import h5py
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "DarthReca/california_burned_areas"

path = hf_hub_download(REPO, filename="raw/patched/512x512.hdf5", repo_type="dataset")
meta_path = hf_hub_download(REPO, filename="metadata.parquet", repo_type="dataset")

with h5py.File(path, "r") as f:
    keys = list(f.keys())
    print(f"top-level groups : {len(keys)}")
    print(f"first five       : {keys[:5]}")

    g = f[keys[0]]
    print(f"\ninside group '{keys[0]}':")
    for k in g:
        d = g[k]
        arr = d[()]
        print(f"  {k:16s} {str(d.shape):20s} {str(d.dtype):10s} "
              f"min={arr.min():>8} max={arr.max():>8}")
    print(f"  attrs: {dict(g.attrs)}")

print("\nmetadata.parquet")
df = pd.read_parquet(meta_path)
print(df.columns.tolist())
print(df.head())
print(f"\nrows: {len(df)}")
for col in df.columns:
    try:
        if df[col].nunique() < 15:
            print(f"  {col}: {sorted(df[col].unique())}")
    except TypeError:
        pass