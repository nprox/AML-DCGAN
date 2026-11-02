#!/usr/bin/env python3
"""Prepare captions and CLIP embeddings for train.py

python models/dataloader.py --nl /datasets/yfcc100m_NL_captions.json \
    --generated yfcc100m_captioned.jsonl --drives-dir /mnt \
    --out /datasets/example --workers 20 --verify-decode

Run  before train.py. Repeating for an existing outdir reuses the prepared tables and
continues interrupted CLIP encoding. Use a new output folder per data configuration.
The pandas merge uses system RAM; Images stay in their own folders. By default, 
discover /mnt/<drive>/yfcc100m. --image-root can supply dataset roots directly instead; 
repeat that argument for multiple roots.
"""
from multiprocessing import get_context
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import numpy as np
import argparse
import hashlib
import torch
import json
import uuid
import os

def save_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def save_table(path,frame):
    table = pa.Table.from_pandas(frame,preserve_index=False)
    tmp=path.with_name(path.name+'.tmp')
    with pa.OSFile(str(tmp),'wb') as f, pa.ipc.new_file(f,table.schema) as writer:
        writer.write_table(table)
    tmp.replace(path)

def open_table(path):
    return pa.ipc.open_file(pa.memory_map(str(path),'r')).read_all()

def caption_list(value):
    if isinstance(value,dict): return list(value.values())
    return value if isinstance(value,list) else [value]

def init_paths(roots):
    global ROOTS
    ROOTS = roots

def check_image(task):
    path,decode = task
    key = Path(path).stem
    paths = [Path(path)]+[root/key[:3]/key[3:6]/(key+'.jpg') for root in ROOTS.get(key[:3],[])]
    for p in paths:
        if not p.is_file(): continue
        try:
            if decode:
                with Image.open(p) as image: image.convert('RGB').load()
            return str(p.resolve())
        except (OSError,ValueError,Image.DecompressionBombError): pass
    return None

def prepare(nl,generated,out,roots=(),workers=12,decode=False,val_fraction=.01,seed=42,drives_dir='/mnt'):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'data.json').is_file(): return json.loads((out/'data.json').read_text())
    frames=[]
    if generated:
        df = pd.read_json(generated,lines=True)[['filepath','captions']]
        df['captions'] = df['captions'].map(caption_list)
        df = df.explode('captions').rename(columns={'filepath':'path','captions':'caption'})
        frames.append(df.assign(source=1))
    if nl:
        df = pd.read_json(nl)[['local_path','caption']].rename(columns={'local_path':'path'})
        frames.append(df.assign(source=0))
    df = pd.concat(frames,ignore_index=True).dropna(subset=['path','caption'])
    del(frames)
    df['caption'] = df['caption'].map(lambda x: ' '.join(x.split()))
    df = df[df['caption'].str.len()>0].copy()
    df['hash'] = df['path'].map(lambda x: Path(x).stem)
    df = df.drop_duplicates(['hash','caption'])
    images = df[['hash','path']].drop_duplicates('hash').reset_index(drop=True)
    root_map={}
    roots = list(map(Path,roots))
    if not roots:
        roots = [drive/'yfcc100m' for drive in Path(drives_dir).iterdir() if (drive/'yfcc100m').is_dir()]
    for root in roots:
        for folder in root.iterdir():
            if folder.is_dir(): root_map.setdefault(folder.name,[]).append(root)
    tasks = ((p,decode) for p in images['path'])
    if workers:
        with get_context('spawn').Pool(workers,initializer=init_paths,initargs=(root_map,)) as pool:
            paths = list(tqdm(pool.imap(check_image,tasks,chunksize=128),total=len(images),desc='Checking images'))
    else:
        init_paths(root_map)
        paths = list(tqdm(map(check_image,tasks),total=len(images),desc='Checking images'))
    images['path'] = paths
    images[images['path'].isna()][['hash']].to_csv(out/'rejected.csv',index=False)
    rejected = int(images['path'].isna().sum())
    images = images.dropna().reset_index(drop=True)
    lookup = pd.Series(images.index.to_numpy(),index=images['hash'])
    df = df[df['hash'].isin(lookup.index)].copy()
    df['image_id'] = df['hash'].map(lookup).astype('int64')
    df = df.sort_values(['image_id','source'],kind='stable').reset_index(drop=True)
    counts = np.bincount(df['image_id'],minlength=len(images))
    natural = np.bincount(df.loc[df['source']==0,'image_id'],minlength=len(images))
    images['end'] = counts.cumsum()
    images['start'] = images['end']-counts
    images['generated'] = images['start']+natural
    # All captions belonging to an image share its split.
    val = images['hash'].map(lambda x: int.from_bytes(hashlib.blake2b(f'{seed}:{x}'.encode(),digest_size=8).digest(),'little')/2**64<val_fraction)
    np.save(out/'train_ids.npy',images.index[~val].to_numpy(dtype='int64'))
    np.save(out/'val_ids.npy',images.index[val].to_numpy(dtype='int64'))
    save_table(out/'images.arrow',images[['path','start','generated','end']])
    save_table(out/'captions.arrow',df[['caption']])
    meta = dict(dataset_id=uuid.uuid4().hex,images={'train':int((~val).sum()),'val':int(val.sum())},
                captions=len(df),rejected=rejected,seed=seed,val_fraction=val_fraction,
                nl=str(nl),generated=str(generated),drives_dir=str(drives_dir),image_roots=list(map(str,roots)))
    save_json(out/'data.json',meta)
    return meta

def embed(data,device,batch_size=512):
    data=Path(data)
    if (data/'embeddings.npy').is_file(): return
    import clip
    table = open_table(data/'captions.arrow')
    partial,progress = data/'embeddings.partial.npy',data/'embedding_progress.json'
    state = {'rows':json.loads(progress.read_text())['rows']} if progress.is_file() else {'rows':0}
    vectors = np.lib.format.open_memmap(partial,mode='r+' if partial.is_file() else 'w+',dtype='float16',shape=(len(table),512))
    model,_ = clip.load('ViT-B/32',device=device,jit=False)
    model.eval().requires_grad_(False)
    model.visual.to('cpu') # encode_text still reads visual.conv1.weight.dtype
    with torch.inference_mode():
        for start in tqdm(range(state['rows'],len(table),batch_size),desc='Encoding captions'):
            texts = table['caption'].slice(start,batch_size).to_pylist()
            batch = clip.tokenize(texts,truncate=True).to(device)
            z = torch.nn.functional.normalize(model.encode_text(batch).float(),dim=-1)
            vectors[start:start+len(texts)] = z.cpu().numpy().astype('float16')
            vectors.flush()
            state.update(rows=start+len(texts))
            save_json(progress,state)
    del(vectors)
    save_json(data/'embeddings.json',dict(**state,model='ViT-B/32',normalized=True,dim=512,dtype='float16'))
    partial.replace(data/'embeddings.npy')
    progress.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nl',type=Path)
    parser.add_argument('--generated',type=Path)
    parser.add_argument('--drives-dir',type=Path,default=Path('/mnt'))
    parser.add_argument('--image-root',type=Path,action='append',default=[])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--verify-decode',action='store_true')
    parser.add_argument('--val-fraction',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(12)
    meta = prepare(args.nl,args.generated,args.out,args.image_root,args.workers,args.verify_decode,args.val_fraction,args.seed,args.drives_dir)
    embed(args.out,args.device,args.batch_size)
    print(json.dumps(meta,indent=2))
    print(f'Ready: python train.py --data {args.out} --model-dir /path/to/local/dinov3 --run runs/eml')

if __name__=='__main__': main()
