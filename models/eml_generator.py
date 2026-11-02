from safetensors.torch import save_file
from safetensors import safe_open
from torch import nn
import torch
import json
import math

"""
Model: ChatGPT 6.1 Sol Extra-High
Can you provide an extension of RMS Norm for the complex numbers? Give a proper citation to any papers cited. Provide the implementation in pytorch.

\begin{response}
citation: 
Zhang, B., & Sennrich, R. (2019).
Root Mean Square Layer Normalization.
Advances in Neural Information Processing Systems, 32.
https://arxiv.org/abs/1910.07467

Contribution:
The cited paper introduces real-valued RMSNorm. This response
proposes a complex-valued extension using squared complex
magnitudes, with normalization across the channel dimension.
The extension and implementation are generated adaptations.
"""
class ComplexRMSNorm(nn.Module):
    def __init__(self, eps=1e-6, dim=1):
        super().__init__()
        if eps <= 0: raise ValueError('eps must be positive')
        self.eps, self.dim = eps, dim

    def forward(self,x):
        # Rescale before squaring: even finite complex64 values can overflow |x|².
        scale = torch.maximum(x.real.abs(), x.imag.abs()).amax(self.dim, keepdim=True).detach().clamp_min(1)
        z = x / scale
        power = z.real.square() + z.imag.square()
        return z * torch.rsqrt(power.mean(self.dim, keepdim=True) + (math.sqrt(self.eps)/scale).square())
    
# \end{response}

# batch, channels, n, m | cat adds to channels
def cat(*xs): return torch.cat(xs,dim=1)
class EMLConv(nn.Module):
    def __init__(self, in_channels, out_channels, log_channels=None, transpose=False, \
                 dtype=torch.complex64, device=None, gain=.2, eps=1e-6, k=None, s=None, p=None, 
                 exp_limit=12., log_floor=1e-3, **kw):
        super().__init__()
        self.exp_limit, self.log_floor = exp_limit, log_floor
        if log_floor is not None: self.log_shift = math.log(math.expm1(math.e-log_floor)) - math.e
        conv = nn.ConvTranspose2d if transpose else nn.Conv2d
        k=(4 if transpose else 3) if k is None else k
        s=(2 if transpose else 1) if s is None else s
        p=1 if p is None else p
        kw = dict(kernel_size=k, stride=s, padding=p, bias=True, dtype=dtype, device=device)
        self.exp_filter = conv(in_channels, out_channels, **kw)
        self.log_filter = conv(in_channels if log_channels is None else log_channels, out_channels, **kw)
        self.norm = ComplexRMSNorm(eps)
        with torch.no_grad():
            self.exp_filter.weight.mul_(gain)
            self.log_filter.weight.mul_(gain)
            self.exp_filter.bias.zero_()
            self.log_filter.bias.fill_(math.e)

    def forward(self, x:torch.tensor, log_input:torch.tensor=None):
        dtype = self.exp_filter.weight.dtype
        y = x if log_input is None else log_input
        u = self.exp_filter(x.to(dtype=dtype))
        v = self.log_filter(y.to(dtype=dtype))
        if self.exp_limit is not None: u = torch.complex((lambda L, x: L*torch.tanh(x/L))(self.exp_limit, u.real), u.imag)
        if self.log_floor is not None: v = torch.complex((lambda L, x, s: L + nn.functional.softplus(x+s))(self.log_floor, v.real, self.log_shift), v.imag)
        return self.norm(torch.exp(u)-torch.log(v))

class ConvEmbedding(nn.Module):
    def __init__(self, v_channels, out_channels, **kw):
        super().__init__()
        temp = 3*v_channels
        self.first = nn.ModuleDict({
            'NN': EMLConv(temp, out_channels, **kw),
            'NV': EMLConv(temp, out_channels, log_channels=v_channels, **kw),
            'VN': EMLConv(v_channels, out_channels, log_channels=temp, **kw),
            'VV': EMLConv(v_channels, out_channels, **kw), # out_channels = 1? or would Conv2D w/ 3 filters have better RGB variability; keeping 3 + no exception for self.second
        })
        self.second = nn.ModuleDict({
            layer: EMLConv(inp_mod*out_channels, out_channels, **kw)
            for layer, inp_mod in (('V3N', 2), ('V3NNV', 3), ('V3NNVNN', 4))
        })
        self.third = nn.ModuleDict({
            layer: EMLConv(inp_mod*out_channels, out_channels, **kw)
            for layer, inp_mod in (('NV', 2), ('NVNN', 3))
        })
        self.fourth = EMLConv(2*out_channels, out_channels, **kw)

    def forward(self, n, v):
        n2 = self.first['NN'](n)
        nv = self.first['NV'](n, v) # exp(Noise) - ln(embedding_vector)
        vn = self.first['VN'](v, n) # opposite
        v2 = self.first['VV'](v)
        v3n = self.second['V3N'](cat(v2, vn))
        v3n2v = self.second['V3NNV'](cat(v2, vn, nv))
        n3n2vn2 = self.second['V3NNVNN'](cat(v2, vn, nv, n2))
        nv2 = self.third['NV'](cat(n3n2vn2, v3n2v))
        nvn2 = self.third['NVNN'](cat(n3n2vn2, v3n2v, v3n))
        return self.fourth(cat(nv2, nvn2))

class ColorFocus(nn.Module):
    def __init__(self, color_channels, out_channels, **kw):
        super().__init__()
        # initially thought of splitting color_channels into r, g, b but input is still abstract so an additional 3->1 won't hurt + more learned activation functions
        # Abstract for color in range(color_channels)
        # double for loop: 2nd order, triple 3rd -> add n-order before terniquete to all then add stride to selection ?
        self.colors = nn.ModuleDict({color: EMLConv(color_channels, out_channels, **kw) for color in 'RGB'})
        self.pair_colors = nn.ModuleDict({color: EMLConv(2*out_channels, out_channels, **kw) for color in ('RG', 'RB', 'GB')})
        self.second_order = nn.ModuleDict({color: EMLConv(2*out_channels, out_channels, **kw) for color in ('RGGB', 'RBBG', 'BRRG')})
        self.all = EMLConv(3*out_channels, out_channels, **kw)
        self.to_color = nn.ModuleDict({color: EMLConv(4*out_channels, out_channels, **kw) for color in 'rgb'})

    def forward(self, x:torch.tensor):
        r, g, b = (self.colors[c](x) for c in 'RGB')
        rg = self.pair_colors['RG'](cat(r, g))
        rb = self.pair_colors['RB'](cat(r, b))
        gb = self.pair_colors['GB'](cat(g, b))
        # RGGB, RBBG, BRRG | independent of order eml(cat(a,b)) = eml(cat(b, a))
        rggb = self.second_order['RGGB'](cat(rg, gb))
        rbbg = self.second_order['RBBG'](cat(rb, gb))
        brrg = self.second_order['BRRG'](cat(rb, rg))
        all = self.all(cat(rggb, rbbg, brrg))

        # built in skip connection | use without repeating middle; skip 2, force second order eml ?
        R = self.to_color['r'](cat(r, rggb, rbbg, all))
        B = self.to_color['b'](cat(b, rggb, brrg, all))
        G = self.to_color['g'](cat(g, rbbg, brrg, all))

        return cat(R, B, G)

class EMLUpcycle(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, start_size, out_size, clip_dim=None, device='cuda', dtype=torch.complex64, **kw):
        super().__init__()

        film = []
        blocks = [ColorFocus(in_channels, mid_channels//3, device=device, **kw)]
        if clip_dim: film.append(FiLM(clip_dim, 3*(mid_channels//3), True, device)) 
        curr_size = start_size
        while 2*curr_size < out_size: # not the final convolution -> up
            blocks.append(nn.Sequential(EMLConv(3*(mid_channels//3), 3*(mid_channels//3), transpose=True, device=device, dtype=dtype, **kw), 
                                        ColorFocus(3*(mid_channels//3), mid_channels//3, device=device, dtype=dtype, **kw)))
            if clip_dim: film.append(FiLM(clip_dim, 3*(mid_channels//3), True, device)) 
            curr_size *= 2
        blocks.append(EMLConv(3*(mid_channels//3), out_channels, transpose=True, device=device, dtype=dtype, **kw))
        if clip_dim: film.append(FiLM(clip_dim, out_channels, True, device)) 
        self.layers = nn.Sequential(*blocks)
        self.films = nn.ModuleList(film) if clip_dim else None

    def forward(self, x:torch.tensor, v=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.films is not None: x = self.films[i](x, v)
        return x

class CRConvUpcycle(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, start_size, out_size, clip_dim=None, device=None, k=4,s=2,p=1, **kw):
        super().__init__()
        # real + complex components
        blocks = [nn.Sequential(nn.Conv2d(2*in_channels, mid_channels, 3,1,1, device=device, **kw),
                                nn.BatchNorm2d(mid_channels, device=device), nn.ReLU(inplace=True))]
        curr_size = start_size
        while 2*curr_size < out_size:
            blocks.append(nn.Sequential(nn.ConvTranspose2d(mid_channels, mid_channels, k,s,p, device=device,**kw), 
                          nn.BatchNorm2d(mid_channels, device=device, **kw), nn.ReLU(inplace=True)))
            curr_size *= 2
        self.films = nn.ModuleList(FiLM(clip_dim, mid_channels, device=device, **kw) for _ in blocks) if clip_dim else None
        self.last = nn.ConvTranspose2d(mid_channels, out_channels, k,s,p, device=device, **kw)
        self.layers = nn.ModuleList(blocks)

        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
                nn.init.normal_(m.weight,0,.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight,1,.02)
                nn.init.zeros_(m.bias)

    def forward(self, x:torch.tensor, v=None): 
        x = cat(x.real, x.imag)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.films is not None: x = self.films[i](x, v)
        return self.last(x).tanh()

class FiLM(nn.Module):
    #Adapted from Perez et al. (2018), https://arxiv.org/abs/1709.07871.
    def __init__(self, clip_dim, channels, complex_features=False, device=None, **kw):
        super().__init__()
        # f(x) = \alpha:gain * x + \beta:complex or real
        self.complex_features, self.clip_dim = complex_features, clip_dim
        self.affine = nn.Linear(clip_dim, channels*(3 if complex_features else 2), device=device, **kw)
        self.w = nn.Parameter(self.affine.weight.new_tensor(3.))
        self.relu6 = nn.ReLU6()
        nn.init.normal_(self.affine.weight, 0, .02)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, v):    
        if v is None: return x
        # want (b, 1, c) \to (b, c) \to (b, 3ch, 1, 1) \to (b, ch, 1, 1); complex | , 1, 1) broadcasts to x: (b, ch, n, v)
        ab = self.affine(v.reshape(-1, self.clip_dim).to(self.affine.weight)).unsqueeze(-1).unsqueeze(-1).chunk(3 if self.complex_features else 2, dim=1)
        # \alpha = f_c(x_i) -> differs on batch | f_c = linear w/ 1 + (.1+relu6(w)/6*.8)*tanh(x) activation, .tanh() applied to real components b, ch indexing
        return x*(1+(.1+self.relu6(self.w)/6*.8)*ab[0].tanh()) + (torch.complex(ab[1], ab[2]) if self.complex_features else ab[1])

class Generator(nn.Module):
    def __init__(self, device=None, clip_dim=16*32, embed_size=16, embed_channels=1*4, embed_width=3*8, color_width=1*8, eml_mid_channels=12*4, eml_channels=3*8,
                 eml_size=128, conv_mid_channels=64, image_size=512, out_channels=3, exp_limit=12., log_floor=1e-3, dtype=torch.float32, **kw):
        super().__init__()
        real_kw = dict(device=device,dtype=dtype, **kw)
        complex_kw = dict(device=device,dtype=torch.complex64 if dtype==torch.float32 else torch.complex128, 
                           exp_limit=exp_limit, log_floor=log_floor, **kw)
        self.config = dict(clip_dim=clip_dim, embed_size=embed_size, embed_channels=embed_channels, embed_width=embed_width,
                           color_width=color_width, eml_mid_channels=eml_mid_channels, eml_channels=eml_channels,
                           eml_size=eml_size, conv_mid_channels=conv_mid_channels, image_size=image_size,
                           out_channels=out_channels, exp_limit=exp_limit, log_floor=log_floor,)
        
        self.embed_size = embed_size
        self.embed_channels = embed_channels

        self.dense = nn.Linear(clip_dim, embed_channels*embed_size**2, **real_kw)
        self.embed = ConvEmbedding(embed_channels, embed_width, **complex_kw)
        self.color = ColorFocus(embed_width, color_width, **complex_kw)
        self.color_film = FiLM(clip_dim, 3*color_width, True, device=device, dtype=dtype)
        self.eml_upcycle = EMLUpcycle(3*color_width, eml_mid_channels, eml_channels, embed_size, eml_size, clip_dim=clip_dim, **complex_kw)
        self.upcycle = CRConvUpcycle(eml_channels, conv_mid_channels, out_channels, eml_size, image_size, clip_dim=clip_dim, k=4,p=1,s=2, **real_kw)

    def sample_noise(self, batch_size, generator=None):
        x = self.dense.weight
        return torch.randn(batch_size, 3*self.embed_channels, self.embed_size, self.embed_size, 
                           device=x.device, dtype=x.dtype, generator=generator)

    def forward(self, x:torch.Tensor, generator=None):
        h = self.dense(x.to(self.dense.weight)).reshape(-1, self.embed_channels, self.embed_size, self.embed_size)
        h = self.embed(self.sample_noise(x.shape[0], generator=generator), h)
        h = self.color_film(self.color(h), x)
        h = self.eml_upcycle(h, x)
        return self.upcycle(h, x)

    '''
    Model: ChatGPT 6.1 Sol Extra-High
    Can you provide methods to save and load model weights as safetensors. 
    \begin{response}
    '''
    def save_weights(self,path):
        state,complex_keys={},[]
        for k,v in self.state_dict().items():
            v=v.detach().cpu().resolve_conj()
            if v.is_complex():
                complex_keys.append(k)
                v=torch.view_as_real(v)
            state[k]=v.contiguous().clone()
        save_file(state,str(path),metadata={"format": 'eml_generator.v1',
                                            "config": json.dumps(self.config), 
                                            "complex_keys":json.dumps(complex_keys)})

    def load_weights(self,path,strict=True):
        with safe_open(str(path),framework="pt",device="cpu") as f:
            complex_keys=set(json.loads((f.metadata() or {}).get("complex_keys","[]")))
            state={}
            for k in f.keys():
                v=f.get_tensor(k)
                state[k]=torch.view_as_complex(v.contiguous()) if k in complex_keys else v
        return self.load_state_dict(state,strict=strict)
    # \end{response}