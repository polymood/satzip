"""Reproduce the single-recording control using the first validation WAV."""
import argparse

import torch

from neuralink import DATA, REPORT, split
from satzip import audio


def main(device):
    src = split()[1][0]
    wav = src.read_bytes()
    out = DATA.parent / "memorized"
    out.mkdir(exist_ok=True)
    general = torch.load(REPORT / "c128-h512-d3.pt", weights_only=False)
    fitted = audio.train([wav], general["table"], **general["config"],
                         steps=6000, batch=2048, device=device)
    fitted["state"] = {key: value.half() for key, value in fitted["state"].items()}
    torch.save(fitted, out / "memorized-weights.pt")
    for name, prior in (("memorized", fitted), ("general", general)):
        stream = audio.compress(wav, prior)
        decoded = audio.decompress(stream, prior)
        if decoded != wav:
            raise ValueError(f"{name} round trip failed")
        (out / f"{src.stem}.{name}.nz").write_bytes(stream)
        if name == "memorized":
            (out / f"{src.stem}.decoded.wav").write_bytes(decoded)
        print(name, len(stream), "bytes, exact decode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args().device)
