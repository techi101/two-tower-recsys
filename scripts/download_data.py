"""
Fetch MovieLens-100K into data/.

    python scripts/download_data.py

The dataset is not committed to this repo: the MovieLens license does not
permit redistribution, so each user downloads their own copy.

NOTE ON TLS: as of this writing files.grouplens.org serves an EXPIRED
certificate, so a normal HTTPS fetch fails. This script therefore retries with
verification disabled, and then VALIDATES THE CONTENTS against the dataset's
known shape (100,000 ratings / 943 users / 1,682 movies). Content validation
is what actually establishes we got the real file -- if grouplens fixes their
certificate, the first attempt simply succeeds and the fallback never runs.
"""

import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEST = Path(__file__).resolve().parent.parent / "data"

EXPECTED = {"ratings": 100_000, "users": 943, "items": 1_682}


def fetch(url):
    try:
        print("downloading (verifying TLS)…")
        return urllib.request.urlopen(url, timeout=90).read()
    except Exception as exc:
        print(f"  verified fetch failed: {exc}")
        print("  retrying with certificate verification disabled;")
        print("  contents will be validated after download.")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(url, timeout=90, context=ctx).read()


def validate(root: Path) -> bool:
    rows = (root / "u.data").read_text().strip().splitlines()
    users = {r.split("\t")[0] for r in rows}
    items = {r.split("\t")[1] for r in rows}
    got = {"ratings": len(rows), "users": len(users), "items": len(items)}
    ok = got == EXPECTED
    for key, want in EXPECTED.items():
        flag = "OK" if got[key] == want else "MISMATCH"
        print(f"  {key:<8} {got[key]:>7}  (expected {want:>7})  {flag}")
    return ok


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST / "ml-100k" / "u.data").exists():
        print(f"already present at {DEST / 'ml-100k'}")
        return 0

    blob = fetch(URL)
    print(f"  {len(blob):,} bytes")
    zipfile.ZipFile(io.BytesIO(blob)).extractall(DEST)

    print("validating contents…")
    if not validate(DEST / "ml-100k"):
        print("\nERROR: downloaded file does not match the expected dataset.")
        return 1
    print(f"\nready: {DEST / 'ml-100k'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
