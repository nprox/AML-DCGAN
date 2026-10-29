from torch.nn.utils.parametrizations import spectral_norm # https://www.youtube.com/watch?v=8ZqFe8goiBw
from eml_generator import EMLConv, cat, FiLM
from safetensors.torch import save_file
from torchvision import transforms
from transformers import AutoModel
from safetensors import safe_open
from pathlib import Path
import torch.nn as nn
import torch
import json
import math
import os

class DynoEmbed(nn.Module):
    def __init__(self, img_channels, embed_channels, out_channels, **kw):
        super().__init__()
        self.First = nn.ModuleDict({
            'VV': EMLConv(embed_channels, out_channels, **kw),
            'VI': EMLConv(embed_channels, out_channels, log_channels=img_channels, **kw),
            'IV': EMLConv(img_channels, out_channels, log_channels=embed_channels, **kw),
            'II': EMLConv(img_channels, out_channels, **kw)
        })
        self.Second = nn.ModuleDict({
            'V3I': EMLConv(3*out_channels, out_channels, **kw), 
            'I3V': EMLConv(3*out_channels, out_channels, **kw),
            'IIVV': EMLConv(4*out_channels, out_channels, **kw),
        })
        self.Third = nn.ModuleDict({
            'V4I4': EMLConv(2*out_channels, out_channels, **kw),
            'V6I6': EMLConv(3*out_channels, out_channels, **kw)
        })
        self.Fourth = EMLConv(2*out_channels, out_channels, **kw)

    def forward(self, img, cap):
        vv = self.First['VV'](cap)
        vi = self.First['VI'](cap, img)
        iv = self.First['IV'](img, cap)
        ii = self.First['II'](img)
        v3i = self.Second['V3I'](cat(vv, vi, iv))
        i3v = self.Second['I3V'](cat(ii, iv, vi))
        i2v2 = self.Second['IIVV'](cat(vv, vi, iv, ii))
        v4i4 = self.Third['V4I4'](cat(v3i, i3v))
        v6i6 = self.Third['V6I6'](cat(v3i, i3v, i2v2))
        return self.Fourth(cat(v4i4, v6i6))

class EMLSkip(nn.Module):
    def __init__(self, in_channels, out_channels, func, **kw):
        super().__init__()
        self.func = func(in_channels, in_channels, out_channels,**kw)
        self.eml = EMLConv(out_channels,out_channels, log_channels=in_channels,**kw)

    def forward(self,inp):
        return self.eml(self.func(inp,inp),inp)
    
class CRConvDownCycle(nn.Module):
    def __init__(self, in_channels, mid_channels=48, out_channels=1, start_size=16, device=None, dtype=torch.float32):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        layers=[]
        curr_size = start_size        
        channels, width = 2*in_channels, mid_channels
        while (curr_size > 2):
            layers.extend([spectral_norm(nn.Conv2d(channels,width,3,2,1,**kw)), nn.LeakyReLU(.2)])
            channels, width, curr_size = width, 2*width, curr_size//2
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(spectral_norm(nn.Conv2d(channels,out_channels,1, **kw)))
        self.layers=nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(cat(x.real,x.imag)).flatten(1)
    
class Discriminator(nn.Module):
#    def __init__(self, model_dir=(Path(os.getcwd()).parent/'datasets'), device=None, embed_channels=1, sync_out_channels=384//4, img_channels=384):
    def __init__(self, model_dir=(Path(os.getcwd()).parent/'datasets'), device=None, embed_channels=8, sync_out_channels=32,
                img_channels=384, feature_channels=48, skip_channels=16, clip_dim=512, dtype=torch.float32, exp_limit=12., log_floor=1e-3):
        super().__init__()
        real_kw = dict(device=device, dtype=dtype)
        complex_kw = dict(device=device, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128, 
                          exp_limit=exp_limit, log_floor=log_floor)
        
        self.clip_dim = clip_dim
        self.config = dict(
            embed_channels=embed_channels, sync_out_channels=sync_out_channels,
            img_channels=img_channels, feature_channels=feature_channels, skip_channels=skip_channels, 
            clip_dim=clip_dim, exp_limit=exp_limit, log_floor=log_floor)
        
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        encoder = getattr(model, 'model', model)
        self.DynoV3 = nn.Sequential(*list(encoder.stages)[:3])
        self.DynoV3.requires_grad_(False).eval().to(**real_kw)

        self.register_buffer('image_mean', torch.tensor([.485, .456, .406], **real_kw).view(1,3,1,1))
        self.register_buffer('image_std', torch.tensor([.229, .224, .225], **real_kw).view(1,3,1,1))

        self.ImageProjection = nn.Sequential(spectral_norm(nn.Conv2d(img_channels, feature_channels, 1, **real_kw)), 
                                             nn.GroupNorm(1, feature_channels, **real_kw))
        
        self.Dense = nn.Linear(clip_dim, embed_channels, **real_kw) #(16*32, 16*16, **real_kw)
        self.DynoEmbed = DynoEmbed(feature_channels, embed_channels, sync_out_channels, **complex_kw)

        # Sequential -> Module to add FiLM
        self.EMLSkip = nn.ModuleList(([EMLSkip(sync_out_channels, skip_channels, DynoEmbed, **complex_kw)]+
                                      [EMLSkip(skip_channels, skip_channels, DynoEmbed, **complex_kw) for _ in range(2)]))
        self.films = nn.ModuleList(FiLM(clip_dim, skip_channels, True, **real_kw) for _ in self.EMLSkip)
        self.DownCycle = CRConvDownCycle(skip_channels, **real_kw)
        self.CaptionProjection = spectral_norm(nn.Linear(clip_dim, feature_channels, bias=False, **real_kw))

    def preprocess(self, img): return ((img.to(self.image_mean)+1)/2-self.image_mean)/self.image_std 
    def forward(self, img:torch.tensor, cap:torch.tensor):
        cap = nn.functional.normalize(cap.reshape(img.shape[0], self.clip_dim).to(self.Dense.weight))
        h = self.DynoV3(self.preprocess(img))
        h = self.ImageProjection(h)

        proj = (h.mean((2,3))*self.CaptionProjection(cap)).sum(1, keepdim=True) / math.sqrt(h.shape[1])
        h = self.DynoEmbed(h, self.Dense(cap)[:, :, None, None].expand(-1, -1, *h.shape[-2:]))
        for skip, film in zip(self.EMLSkip, self.films):
            h = film(skip(h), cap)
        return self.DownCycle(h) + proj
    
    def train(self, mode=True):
        super().train(mode)
        self.DynoV3.eval()
        return self
    
    '''
    Model: ChatGPT 6.1 Sol Extra-High
    Can you provide methods to save and load model weights as safetensors. The DynoV3 ConvNext Small weights should not be present in the safetensors
        and loading should still grab from model_dir.
    \begin{response}
    '''
    def save_weights(self,path):
        state,complex_keys={},[]
        for k,v in self.state_dict().items():
            if k.startswith("DynoV3."):continue
            v=v.detach().cpu().resolve_conj()
            if v.is_complex():
                complex_keys.append(k)
                v=torch.view_as_real(v)
            state[k]=v.contiguous().clone()

        save_file(state,str(path),metadata={
                "format": "eml_discriminator.v1",
                "config": json.dumps(self.config),
                "complex_keys": json.dumps(complex_keys)
            })

    def load_weights(self,path,strict=True):
        with safe_open(str(path),framework="pt",device="cpu") as f:
            meta=f.metadata() or {}
            if meta.get("format")!="eml_discriminator.v1":
                raise ValueError("Unsupported discriminator checkpoint format")
            complex_keys=set(json.loads(meta["complex_keys"]))
            state={}
            for k in f.keys():
                if k.startswith("DynoV3."):
                    raise ValueError("Checkpoint unexpectedly contains DINOv3 weights")
                v=f.get_tensor(k)
                state[k]=torch.view_as_complex(v.contiguous()) if k in complex_keys else v

        # Retain the backbone loaded by __init__ from model_dir.
        state.update({k:v for k,v in self.state_dict().items()
                      if k.startswith("DynoV3.")})
        return self.load_state_dict(state,strict=strict)
    # \end{response}