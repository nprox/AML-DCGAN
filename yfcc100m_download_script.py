#!/usr/bin/env python3
"""
Download YFCC image prefixes across drives mounted directly under /mnt.

Requires Python 3 and the AWS CLI on PATH. A .complete marker records a
successful sync. Existing folders without that marker are synced again.
"""

from pathlib import Path
import numpy as np
import subprocess
import argparse
import shutil
import sys
import re
import os

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drives-dir", type=Path, default="/mnt")
    args = parser.parse_args()

    yfcc100m = []
    drives_dir = Path(args.drives_dir)
    for drive in drives_dir.iterdir():
        drive_dir = drives_dir / re.match(r'.*/([^/]+)$', str(drive)).group(1)
        if not os.path.exists(drive_dir / 'yfcc100m'):
            os.mkdir(drive_dir / 'yfcc100m')
        drive_dir = drive_dir / 'yfcc100m'
        yfcc100m.extend([re.match(r'.*/([^/]+)$', str(fldr)).group(1) for fldr in drive_dir.iterdir() if (fldr/'.complete').is_file()])

    folders = subprocess.run(["aws", "s3", "ls", "s3://multimedia-commons/data/images/", "--no-sign-request", "--region", "us-west-2",], \
                            capture_output=True, text=True, check=True,).stdout.splitlines()
    folders = [fdr for line in folders if (fdr := re.match(r".*PRE (.+)/", line).group(1)) not in yfcc100m]
    indices = np.random.permutation(np.arange(folders.__len__()))

    cntr = 0
    for drive in drives_dir.iterdir():
        drive_dir = drives_dir / re.match(r'.*/([^/]+)$', str(drive)).group(1) / 'yfcc100m'
        for idx in indices[cntr:].copy():
            if shutil.disk_usage(drive_dir)[-1] < 5*1024**3:
                break
            print(folders[idx])
            dwnl = subprocess.run(["aws", 's3', 'sync', f's3://multimedia-commons/data/images/{folders[idx]}/',
                                (drive_dir/folders[idx]), '--no-sign-request', '--region', 'us-west-2',],
                                check = True,)
            (drive_dir/folders[idx]/'.complete').touch()
            cntr += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the script to retry unfinished downloads.", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Download stopped: {exc}", file=sys.stderr)
        sys.exit(1)