"""Download the test data used by the notebooks into data/.

    uv run python scripts/download_data.py            # everything (about 410 MB)
    uv run python scripts/download_data.py --no-lst   # Sentinel samples only (about 50 MB)

Sources need no account:
- Sentinel-1, -2, -3 windows: Microsoft Planetary Computer (via satzip.fetch).
- ESA CCI LST, SLSTR-B, 2023-01-01 daytime: CEDA archive.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import satzip

DATA = Path(__file__).resolve().parents[1] / "data"
SUMMER = "2024-06-01T00:00:00Z/2024-09-30T23:59:59Z"
CLEAR = {"eo:cloud_cover": {"lt": 5}}
S2 = ["B02", "B03", "B04", "B08"]

SENTINEL = [  # same arguments as notebooks/01_copernicus.py, so the cache names match
    ("sentinel-2-l2a", 1.44, 43.60, S2, 512, SUMMER, CLEAR),  # Toulouse
    ("sentinel-2-l2a", 1.44, 43.60, S2, 2048, SUMMER, CLEAR),  # Toulouse, large sample for big models
    ("sentinel-2-l2a", 1.60, 48.25, S2, 512, SUMMER, CLEAR),  # Beauce farmland
    ("sentinel-2-l2a", 2.35, 48.85, S2, 512, SUMMER, CLEAR),  # Paris (prior training)
    ("sentinel-2-l2a", 4.83, 45.76, S2, 512, SUMMER, CLEAR),  # Lyon (prior training)
    ("sentinel-2-l2a", -1.55, 47.22, S2, 512, SUMMER, CLEAR),  # Nantes (prior training)
    ("sentinel-1-grd", 1.44, 43.60, ["vv", "vh"], 512),
    ("sentinel-3-olci-wfr-l2-netcdf", -5.0, 47.0, [f"oa0{i}-reflectance" for i in range(3, 9)], 512,
     "2024-07-01T00:00:00Z/2024-07-31T23:59:59Z"),
]

LST_NAME = "ESACCI-LST-L3C-LST-SLSTRB-0.01deg_1DAILY_DAY-20230101000000-fv4.00.nc"
LST_URL = ("https://dap.ceda.ac.uk/neodc/esacci/land_surface_temperature/data/SENTINEL3B_SLSTR/L3C/0.01/"
           f"v4.00/daily/2023/01/01/{LST_NAME}")


def download(url: str, dest: Path) -> None:
    """Stream to dest.part, check the size against the server, then rename."""
    if dest.exists():
        print(f"ok      {dest.name}")
        return
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, part.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r        {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if total and part.stat().st_size != total:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {total}")
    part.rename(dest)
    print(f"fetched {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-lst", action="store_true", help="skip the 359 MB ESA CCI LST file")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for spec in SENTINEL:
        img, _ = satzip.fetch(*spec, cache=DATA)
        print(f"ok      {spec[0]} ({spec[1]}, {spec[2]}): {img.shape} {img.dtype}")
    if not args.no_lst:
        download(LST_URL, DATA / LST_NAME)


if __name__ == "__main__":
    main()
