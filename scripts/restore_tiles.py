"""Restore the original downloader NPZ files from a benchmark bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

import satzip


def restore(bundle, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle) as z:
        for row in json.loads(z.read("manifest.json")):
            name = row["file"]
            if Path(name).name != name or not name.endswith(".npz"):
                raise ValueError(f"Invalid tile filename: {name}")
            if "crop_of" in row:
                source = row["crop_of"]
                if Path(source).name != source:
                    raise ValueError(f"Invalid source filename: {source}")
                y, x, h, w = row["window"]
                with np.load(dest / source) as tile:
                    img = tile["img"][:, y:y+h, x:x+w]
            else:
                payload = z.read(row["payload"])
                if row["payload"].endswith(".sz"):
                    img = satzip.decompress(payload)
                else:
                    with np.load(io.BytesIO(payload)) as archive:
                        prior = {k: torch.from_numpy(archive[k]).float() for k in archive.files if k != "stream"}
                        img = satzip.decompress(archive["stream"].tobytes(), prior=prior)
            fill = np.array(row["fill"], dtype=row["fill_dtype"])
            result = io.BytesIO()
            np.savez(result, img=img, fill=fill)
            if hashlib.sha256(result.getbuffer()).hexdigest() != row["sha256"]:
                raise ValueError(f"Restored file differs from the original: {name}")
            with (dest / name).open("xb") as f:
                f.write(result.getbuffer())
            print(name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    restore(args.bundle, args.output_directory)
