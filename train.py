#!/usr/bin/env python3
"""Train the repository's Generator and Discriminator using dataloader.py's cache.
python train.py --data /home/nprox/datasets/eml_gen --model-dir /home/nprox/datasets \
    --run runs/eml_simple --batch-size 95 --epochs 100

Adam betas=(0,.9), hinge loss, mismatched captions, gradient clipping at 10.
Each epoch saves model/Adam/RNG safetensors, JSON metrics, a CSV and a preview.
Rerun with the same --run to resume the last complete epoch; --epochs is the total.
Reuse the same prepared data and local DINOv3 backbone. Interrupted epochs restart.
Choose batch size manually. Real/complex layers use float32/complex64.

New .venv, Python 3.11/3.12:
python -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install pandas pyarrow==23.0.1 Pillow transformers==4.57.1 safetensors tqdm ftfy regex 'setuptools<81' wheel
python -m pip install --no-build-isolation --no-deps git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
"""
from models.dataloader import open_table, save_json
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import numpy as np
from models.eml_discriminator import Discriminator
from models.eml_generator import Generator
from torch.nn import functional as F
from torchvision.utils import save_image
from safetensors.torch import save_file
from safetensors import safe_open
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
import torch
import json
import time
import csv
import os
"""
Model: ChatGPT 6.1 Sol Extra-high

Inspect https://github.com/nprox/AML-DCGAN for the generator/discriminator architecture and interfaces.
Additional context: <dataloader.py> prepares image/caption tables, image-level train/validation splits, 
and cached CLIP embeddings. Images remain in their original folders. Use the supplied script’s output 
format and helpers. Write one compact, executable train.py containing the training dataset and batching helpers. 
Target cuda:0 with configurable batch size and multiprocessing. Use hinge loss and Adam, 
include mismatched-caption negatives, and handle frozen components correctly.

Save atomic checkpoints every epoch containing model weights, optimizer/RNG states, configuration, and progress, 
using safetensors and JSON. Automatically resume the latest completed epoch. Save retrievable training metrics, 
image-read errors, and repeatable validation previews.
\begin{response}
"""

class CaptionDataset(Dataset):
    def __init__(self,data,split='train',size=256,seed=42,generated_probability=.5):
        self.data=Path(data)
        self.split, self.size, self.seed = split, size, seed
        self.generated_probability = generated_probability
        self.n = json.loads((self.data/'data.json').read_text())['images'][split]
        self.images=self.ids=self.emb=None

    def __len__(self): return self.n
    def __getstate__(self): return dict(self.__dict__,images=None,ids=None,emb=None)
    def _open(self):
        if self.images is None:
            self.images = open_table(self.data/'images.arrow')
            self.ids = np.load(self.data/f'{self.split}_ids.npy',mmap_mode='r')
            self.emb = np.load(self.data/'embeddings.npy',mmap_mode='r')

    def __getitem__(self,key):
        idx,epoch = key if isinstance(key,tuple) else (key,0)
        self._open()
        ident = int(self.ids[idx])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,epoch,ident]))
        path = self.images['path'][ident].as_py()
        start,mid,end = (self.images[k][ident].as_py() for k in ('start','generated','end'))
        source = int(mid==start or (mid<end and rng.random()<self.generated_probability))
        cap_id = int(rng.integers(mid,end) if source else rng.integers(start,mid))
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
                x = TF.pil_to_tensor(image).float().div_(127.5).sub_(1)
        except (OSError,ValueError,Image.DecompressionBombError) as e:
            return dict(error=str(e),path=path,id=ident)
        return dict(image=x,embedding=torch.from_numpy(np.array(self.emb[cap_id],dtype=np.float32)).unsqueeze(0),
                    id=ident,caption_id=cap_id,source=source)

class EpochSampler(Sampler):
    def __init__(self,n,seed=42,shuffle=True):
        self.n, self.seed, self.shuffle, self.epoch = n, seed, shuffle, 0
    def __len__(self): return self.n
    def __iter__(self):
        ids = np.random.default_rng(self.seed+self.epoch).permutation(self.n) if self.shuffle else range(self.n)
        for i in ids: yield int(i),self.epoch if self.shuffle else 0

def collate(rows):
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    batch = dict(errors=errors,n=len(rows))
    if rows:
        for k in ('image','embedding'): batch[k]=torch.stack([r[k] for r in rows])
        for k in ('id','caption_id','source'): batch[k]=torch.tensor([r[k] for r in rows])
    return batch

def worker_init(_): torch.set_num_threads(1)
def loader(dataset,batch_size,workers,shuffle=True):
    sampler = EpochSampler(len(dataset),dataset.seed,shuffle)
    kw = dict(batch_size=batch_size,sampler=sampler,num_workers=workers,collate_fn=collate,
              pin_memory=torch.cuda.is_available(),worker_init_fn=worker_init,
              generator=torch.Generator().manual_seed(dataset.seed),timeout=120 if workers else 0)
    if workers: kw.update(persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
    return DataLoader(dataset,**kw),sampler


def save_tree(path,value):
    tensors={}
    def encode(x):
        if torch.is_tensor(x):
            key=str(len(tensors))
            v=x.detach().cpu().resolve_conj().resolve_neg()
            is_complex=v.is_complex()
            tensors[key]=(torch.view_as_real(v) if is_complex else v).contiguous().clone()
            return dict(t='tensor',key=key,complex=is_complex)
        if isinstance(x,dict): return dict(t='dict',items=[[encode(k),encode(v)] for k,v in x.items()])
        if isinstance(x,(list,tuple)): return dict(t='tuple' if isinstance(x,tuple) else 'list',items=[encode(v) for v in x])
        return dict(t='value',value=x)
    tree=encode(value)
    save_file(tensors,str(path),metadata={'format':'eml_training.v1','tree':json.dumps(tree)})

def load_tree(path):
    with safe_open(str(path),framework='pt',device='cpu') as f:
        def decode(x):
            if x['t']=='tensor':
                v=f.get_tensor(x['key'])
                return torch.view_as_complex(v.contiguous()) if x['complex'] else v
            if x['t']=='dict': return {decode(k):decode(v) for k,v in x['items']}
            if x['t'] in ('list','tuple'):
                values=[decode(v) for v in x['items']]
                return tuple(values) if x['t']=='tuple' else values
            return x['value']
        return decode(json.loads(f.metadata()['tree']))

def train_step(g,d,og,od,batch,device):
    g.train()
    d.train()
    og.zero_grad(set_to_none=True)
    od.zero_grad(set_to_none=True)
    x,c=(batch[k].to(device,non_blocking=True) for k in ('image','embedding'))
    fake=g(c)
    real_score=d(x,c)
    fake_score=d(fake.detach(),c)
    negative=F.relu(1+fake_score).mean()
    wrong=c.roll(1,0)
    valid=F.cosine_similarity(c.flatten(1),wrong.flatten(1),dim=1)<.9999
    wrong_score=real_score.new_zeros(())
    if valid.any():
        wrong_score=d(x,wrong)[valid]
        negative=(negative+F.relu(1+wrong_score).mean())/2
    d_loss=F.relu(1-real_score).mean()+negative
    d_loss.backward()
    trainable=[p for p in d.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable,10.)
    od.step()
    od.zero_grad(set_to_none=True)
    for p in trainable: p.requires_grad_(False)
    d.eval() # Keep spectral norm buffers fixed; retain gradients through the image.
    try:
        g_loss=-d(fake,c).mean()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(),10.)
        og.step()
    finally:
        for p in trainable: p.requires_grad_(True)
        d.train()
    return dict(d_loss=float(d_loss.detach()),g_loss=float(g_loss.detach()),real_score=float(real_score.detach().mean()),
                fake_score=float(fake_score.detach().mean()),wrong_score=float(wrong_score.detach().mean()),
                mismatch_fraction=float(valid.float().mean()),generated_fraction=float(batch['source'].float().mean()))

def checkpoint(run,epoch,step,g,d,og,od,config,metrics,fixed,device):
    root=run/'checkpoints'
    stage=root/f'epoch_{epoch:04d}.tmp'
    shutil.rmtree(stage,ignore_errors=True)
    stage.mkdir(parents=True)
    g.save_weights(stage/'generator.safetensors')
    d.save_weights(stage/'discriminator.safetensors')
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
    save_tree(stage/'training.safetensors',dict(g_optimizer=og.state_dict(),d_optimizer=od.state_dict(),rng=rng))
    if fixed['n']:
        g.eval()
        with torch.inference_mode():
            images=g(fixed['embedding'].to(device),generator=torch.Generator(device=device).manual_seed(42))
        save_image((torch.cat((fixed['image'],images.cpu()))+1)/2,stage/'preview.png',nrow=fixed['n'])
    save_json(stage/'state.json',dict(epoch=epoch,step=step,config=config,metrics=metrics))
    for path in stage.iterdir():
        with path.open('rb') as f: os.fsync(f.fileno())
    final=root/f'epoch_{epoch:04d}'
    stage.replace(final)
    save_json(root/'latest.json',dict(checkpoint=final.name,epoch=epoch,step=step))
    rows=[json.loads(p.read_text())['metrics'] for p in sorted(root.glob('epoch_*/state.json')) if not p.parent.name.endswith('.tmp')]
    with (run/'epochs.csv.tmp').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (run/'epochs.csv.tmp').replace(run/'epochs.csv')
    return final

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=Path('runs/eml'))
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--image-size',type=int,default=256)
    parser.add_argument('--lr-g',type=float,default=1e-4)
    parser.add_argument('--lr-d',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a=parser.parse_args()
    run,device=a.run,torch.device(a.device)
    run.mkdir(parents=True,exist_ok=True)
    completed=sorted(p for p in (run/'checkpoints').glob('epoch_*/state.json') if not p.parent.name.endswith('.tmp'))
    prior=json.loads(completed[-1].read_text()) if completed else None
    seed=prior['config']['args']['seed'] if prior else a.seed
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if device.type=='cuda': torch.cuda.set_device(device)
    gconf=prior['config']['generator'] if prior else {'image_size':a.image_size}
    dconf=prior['config']['discriminator'] if prior else {}
    g=Generator(device=device,**gconf)
    d=Discriminator(model_dir=a.model_dir,device=device,**dconf)
    og=torch.optim.Adam(g.parameters(),lr=a.lr_g,betas=(0.,.9),foreach=True)
    od=torch.optim.Adam((p for p in d.parameters() if p.requires_grad),lr=a.lr_d,betas=(0.,.9),foreach=True)
    epoch=step=0
    if prior:
        path=completed[-1].parent
        g.load_weights(path/'generator.safetensors')
        d.load_weights(path/'discriminator.safetensors')
        state=load_tree(path/'training.safetensors')
        og.load_state_dict(state['g_optimizer'])
        od.load_state_dict(state['d_optimizer'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(state['rng']['cuda'])
        epoch,step=prior['epoch'],prior['step']
        print(f'Resuming after epoch {epoch}',flush=True)
    batch_size=a.batch_size or (prior['config']['batch_size'] if prior else 2)
    size=g.config['image_size']
    dataset=CaptionDataset(a.data,'train',size,seed)
    dl,sampler=loader(dataset,batch_size,a.workers)
    val=CaptionDataset(a.data,'val',size,seed)
    fixed=collate([val[i] for i in range(min(batch_size,8,len(val)))])
    a.seed,a.batch_size,a.image_size=seed,batch_size,size
    a.lr_g,a.lr_d=og.param_groups[0]['lr'],od.param_groups[0]['lr']
    config=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                generator=g.config,discriminator=d.config,batch_size=batch_size)
    for epoch in range(epoch+1,a.epochs+1):
        sampler.epoch=epoch
        totals={}
        seen=skipped=0
        started=time.perf_counter()
        bar=tqdm(dl,desc=f'Epoch {epoch}')
        for b in bar:
            skipped+=len(b['errors'])
            if b['errors']:
                with (run/'errors.jsonl').open('a') as f:
                    for error in b['errors']: f.write(json.dumps(dict(epoch=epoch,**error))+'\n')
            if not b['n']: continue
            stats=train_step(g,d,og,od,b,device)
            step+=1
            seen+=b['n']
            for k,v in stats.items(): totals[k]=totals.get(k,0.)+v*b['n']
            bar.set_postfix(D=f"{stats['d_loss']:.4f}",G=f"{stats['g_loss']:.4f}",refresh=False)
        if not seen: break
        seconds=time.perf_counter()-started
        metrics=dict({k:v/seen for k,v in totals.items()},epoch=epoch,step=step,images=seen,skipped=skipped,
                     seconds=seconds,images_per_second=seen/seconds,lr_g=og.param_groups[0]['lr'],lr_d=od.param_groups[0]['lr'])
        path=checkpoint(run,epoch,step,g,d,og,od,config,metrics,fixed,device)
        print(f'Saved {path}',flush=True)

if __name__=='__main__': main()