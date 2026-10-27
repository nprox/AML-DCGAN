#!/usr/bin/env python3
"""Caption downloaded YFCC100M images using a local llama-server.

Requires Python 3, NumPy, OpenAI CLIP, and a CUDA-enabled llama-server build on PATH.

Uses multiprocessing to generate multiple caption variants per image.
Processes subfolders within prefixes marked .complete, skips subfolders marked .parsed, 
and appends image metadata and captions to a JSONL file.
Marks each subfolder .parsed after its results are successfully written.
"""
from pathlib import Path
from multiprocessing import get_context
from base64 import b64encode
from urllib.request import Request, urlopen
import numpy as np
import subprocess
import argparse
import json
import time
#import clip
import sys
import re
import os

def sub_captions(task):
    img, level, prmpt = task
    return caption_image((img, {level: prmpt}))

def caption_image(task):
    img, prompt = task
    encoded = b64encode(img.read_bytes()).decode("ascii")
    mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"

    relative_path = Path(*img.parts[img.parts.index("yfcc100m") + 1:])
    batch = {
        "filename": img.name,
        "filepath": str(img),
        "downloadpath": f"s3://multimedia-commons/data/images/{relative_path.as_posix()}",
        "captions": {},
    }

    for level, prmpt in prompt.items():
        if not prmpt['prompt'].strip():
            continue
        payload = {
            "model": "yfcc-captioner",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prmpt['prompt']},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                    }},
                ],
            }],
            "max_tokens": 77,
            "temperature": prmpt['temp'],
            "stream": False,
        }
        request = Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=300) as response:
            choice = json.load(response)["choices"][0]

        caption = (choice["message"].get("content") or "").strip()
        """
        if not caption or choice["finish_reason"] == "length":
            print(caption)
            raise RuntimeError(f"Empty or truncated {level} caption for {img}; request a shorter caption.")
        clip.tokenize(caption, context_length=77, truncate=False)"""
        batch["captions"][level] = caption
    return batch

def reparse_yfcc1000m(fldr_1):
    return [Path(sub_fldr) for sub_fldr in fldr_1.iterdir() if sub_fldr.is_dir() and not (sub_fldr/'.parsed').is_file()]

def save_queue(path,folders):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps([str(p) for p in folders]),encoding="utf-8")
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drives-dir", type=Path, default="/mnt")
    parser.add_argument("--out-dir", type=Path, default=os.getcwd())
    parser.add_argument("--parallel", type=int, default=28)
    parser.add_argument("--model", type=str, default='bartowski/Qwen3.8-27B-GGUF:Q4_K_M')
    parser.add_argument("--folders-per-batch", type=int, default=50)
    parser.add_argument("--reparse",action="store_true")
    parser.add_argument("--stop-early", action="store_true")
    args = parser.parse_args()

    fldr = Path(args.out_dir)
    fldr.mkdir(parents=True, exist_ok=True)
    queue=fldr/"for_captioning.json"
    if args.reparse or not queue.is_file():
        fldrs = [fldr_1 for drive in Path(args.drives_dir.resolve()).iterdir() if (drive/'yfcc100m').is_dir() \
                    for fldr_1 in (drive/'yfcc100m').iterdir() if fldr_1.is_dir() and (fldr_1/'.complete').is_file()]
        
        yfcc100m = []
        with get_context("spawn").Pool(processes=args.parallel) as pool:
            for folders in pool.imap(reparse_yfcc1000m, fldrs, chunksize=1):
                yfcc100m.extend(folders)
        
        indices=np.random.permutation(len(yfcc100m))
        yfcc100m=[yfcc100m[idx] for idx in indices]
        del(fldrs, indices)
    else:
        yfcc100m=[Path(p) for p in json.loads(queue.read_text(encoding="utf-8"))]
        idx=0
        while idx<len(yfcc100m) and (yfcc100m[idx]/".parsed").is_file():
            idx+=1
        yfcc100m = yfcc100m[idx:]
        del(idx)
        
    ### indices before -> parse beginning to remove
    save_queue(queue, yfcc100m)
    if args.stop_early:
        return

    if not yfcc100m:
        print("No unfinished completed folders found.")
        return

    PROMPT = {
        'small': {"prompt": "Describe this image in one concise English sentence, at most 25 words. Mention the main subject, action, and setting when visible. Return only the caption.",
                'temp': .15},
        'medium': {'prompt': "Describe this image in one concise English sentence, at most 50 words at minimum 25 words. Mention the main subject, action, and setting when visible. Return only the caption.",
                    'temp': .2},  
        'large': {'prompt': "Return only a caption", 'temp': .3},
        'wild': {'prompt': 'Return only a caption', 'temp': .9}
        }

    server = subprocess.Popen([
        "llama-server", "-hf", args.model,
        "--n-gpu-layers", "99",
        "--parallel", str(args.parallel),
        "--ctx-size", str(4096 * args.parallel),
        "--cont-batching", "--jinja", "--reasoning", "off",
        "--alias", "yfcc-captioner", "--host", "127.0.0.1", "--port", "8080",
        "--log-verbosity", "2",
    ])

    # batch then add to whole
    try:
        # Loading model.
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {server.returncode}")
            try:
                with urlopen("http://127.0.0.1:8080/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            #print('connecting...') # get's noisy
            time.sleep(0.5)

        (fldr / 'yfcc100m_captioned.jsonl').touch()
        with get_context("spawn").Pool(processes=args.parallel) as pool:
            n = args.folders_per_batch
            for start in range(0, len(yfcc100m), n):
                sources = yfcc100m[start:start+n]
                tasks = ((img, level, prmpt) for source in sources for img in source.iterdir() 
                    if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for level, prmpt in PROMPT.items() if prmpt["prompt"].strip())
                
                batch = {}
                for value in pool.imap_unordered(sub_captions, tasks, chunksize=1):
                    key = value["filepath"]
                    if key in batch:
                        batch[key]["captions"].update(value["captions"])
                    else:
                        batch[key] = value
                
                with (fldr / "yfcc100m_captioned.jsonl").open("a", encoding="utf-8") as file:
                    for value in batch.values():
                        file.write(json.dumps(value, ensure_ascii=False) + "\n")

                for source in sources:
                    (source / '.parsed').touch()
                print(f"Captioned {len(sources)} folders: {len(batch)} images\n{sources}", flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the script to retry unfinished downloads.", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Download stopped: {exc}", file=sys.stderr)
        sys.exit(1)