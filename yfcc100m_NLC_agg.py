#!/usr/bin/env python3
""" Aggregate existing natural language captions and descriptions from downloaded photos, OpenAI's yfcc100m_subset_data.tsv, and yfcc100m_dataset.sql.
Make sure the commands:
    aws s3 cp s3://multimedia-commons/tools/etc/yfcc100m_dataset.sql \
    /home/nprox/datasets/yfcc100m_dataset.sql \
    --no-sign-request --region us-west-2

    wget https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2
    bunzip2 yfcc100m_subset_data.tsv.bz2
have been ran in desired data-dir: (/AML-DCGAN).parent / datasets by default
Requires urllib, BeautifulSoup bs4, pandas, and sqlite3
"""
from urllib.parse import unquote_plus
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import argparse
import sqlite3
import gc
import os

def main():
    datasets = Path(os.getcwd()).parent / Path("datasets")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=datasets)
    parser.add_argument("--out-dir", type=Path, default=datasets)
    parser.add_argument("--drives-dir", type=Path, default="/mnt")
    args = parser.parse_args()

    if args.data_dir == datasets or args.out_dir == datasets:
        datasets.mkdir(exist_ok=True, parents=True)

    data = pd.read_csv(args.data_dir / Path('yfcc100m_subset_data.tsv'), names=['line_number', 'photo_id', 'photo_hash'], header=None, sep='\t')
    folders = data['photo_hash'].str[:3].unique()

    drives_dir = Path(args.drives_dir)
    path_map = {fldr.name: Path(fldr) 
                for drive in drives_dir.iterdir() 
                if (drive/'yfcc100m').is_dir()
                for fldr in (drive/'yfcc100m').iterdir() 
                if fldr.name in folders}
    # all photos .jpg
    dwnld_photos = data['photo_hash'].map(lambda x: path_map.get(x[:3], Path())/Path(x[3:6])/Path(x+'.jpg') if path_map.get(x[:3], False) != False else pd.NA).dropna()
    dwnld_photos = pd.DataFrame(dwnld_photos.to_numpy(), columns=['local_path'])
    #dwnld_photos = dwnld_photos[dwnld_photos.map(Path.exists)] #.map(lambda x: x if x.exists() else pd.NA).dropna()
    dwnld_photos['photo_hash'] = dwnld_photos['local_path'].map(lambda x: x.stem)
    dwnld_photos = pd.merge(dwnld_photos, data[['photo_hash', 'photo_id']], on='photo_hash', how='left')
    del(data, folders, datasets)
    gc.collect()

    con=sqlite3.connect(f'file:{str(args.data_dir)}/yfcc100m_dataset.sql?mode=ro',uri=True)
    con.execute('''
        CREATE TEMP TABLE IF NOT EXISTS photoid (
            photoid INTEGER PRIMARY KEY
        )
    ''')
    con.execute('DELETE FROM temp.photoid')

    con.executemany(
        'INSERT OR IGNORE INTO temp.photoid(photoid) VALUES (?)',
        ((int(x),) for x in dwnld_photos['photo_id'])
    )
    con.commit()

    df = pd.read_sql_query('''
        SELECT y.photoid, y.title, y.description
        FROM temp.photoid AS w
        JOIN main.yfcc100m_dataset AS y ON y.photoid = w.photoid
    ''', con)
    con.close()
    
    dwnld_photos = (lambda x: x[[y for y in x.columns if y != 'photoid']])(pd.merge(dwnld_photos, df, left_on='photo_id', right_on='photoid'))
    del(df)
    gc.collect()

    dwnld_photos['title'] = dwnld_photos['title'].map(unquote_plus)
    dwnld_photos['description'] = dwnld_photos['description'].map(unquote_plus)
    msk = dwnld_photos['description'].str.contains('<|>', regex=True)
    dwnld_photos.loc[msk, 'description'] = dwnld_photos[msk]['description'].map(lambda x: BeautifulSoup(x, 'lxml').get_text(' ', strip=True), na_action='ignore')
    dwnld_photos.loc[~dwnld_photos['title'].str.contains(r'[\.\!\?]$'), 'title'] = dwnld_photos[~dwnld_photos['title'].str.contains(r'[\.\!\?]$')]['title'] + '.'
    dwnld_photos['title'] = dwnld_photos['title'].str.replace(r'\.+\s*$', '. ', regex=True)
    dwnld_photos['caption'] = dwnld_photos['title'] + dwnld_photos['description']

    if not (args.out_dir/Path('yfcc100m_NL_captions.json')).exists():
        dwnld_photos['processed'] = False
    else:
        exists = pd.read_json(args.out_dir/Path('yfcc100m_NL_captions.json'))
        dwnld_photos = pd.merge(dwnld_photos, exists[['photo_id', 'processed']], on='photo_id', how='left')
        dwnld_photos.loc[dwnld_photos['processed'].isna(), 'processed'] = False
    dwnld_photos[['local_path', 'photo_id', 'caption', 'processed']].to_json(args.out_dir/Path('yfcc100m_NL_captions.json'), default_handler=str, force_ascii=False)

if __name__ == "__main__":
    main()