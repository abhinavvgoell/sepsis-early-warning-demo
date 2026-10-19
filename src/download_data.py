"""Fetch the public PhysioNet Challenge 2019 training PSV files from its S3 mirror."""

from __future__ import annotations

import argparse
import concurrent.futures
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET_URL = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "challenge-2019/1.0.0/training/"
NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_patient_files() -> list[str]:
    """List all PSV keys using the anonymous S3 ListObjectsV2 API."""
    keys: list[str] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX}
        if token:
            params["continuation-token"] = token
        url = BUCKET_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        keys.extend(
            item.findtext(f"{NAMESPACE}Key")
            for item in root.findall(f"{NAMESPACE}Contents")
            if (item.findtext(f"{NAMESPACE}Key") or "").endswith(".psv")
        )
        truncated = root.findtext(f"{NAMESPACE}IsTruncated") == "true"
        if not truncated:
            return keys
        token = root.findtext(f"{NAMESPACE}NextContinuationToken")
        if not token:
            raise RuntimeError("S3 listing was truncated but returned no continuation token")


def download_one(key: str, output_dir: Path) -> bool:
    relative = key.removeprefix(PREFIX)
    destination = output_dir / Path(relative)
    if destination.exists() and destination.stat().st_size > 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = BUCKET_URL + urllib.parse.quote(key, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "sepsis-early-warning-project/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if not payload:
                raise IOError(f"Empty download for {key}")
            destination.write_bytes(payload)
            return True
        except (urllib.error.URLError, TimeoutError, IOError):
            if attempt == 2:
                raise
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = list_patient_files()
    if not keys:
        raise RuntimeError("No patient PSV files were listed from the PhysioNet S3 mirror")
    print(f"Found {len(keys):,} patient files. Downloading to {args.output_dir} …", flush=True)
    downloaded = skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, key, args.output_dir) for key in keys]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            if future.result():
                downloaded += 1
            else:
                skipped += 1
            if index % 1000 == 0 or index == len(futures):
                print(f"Processed {index:,}/{len(futures):,} files", flush=True)
    print(f"Done. Downloaded {downloaded:,}; already present {skipped:,}.")


if __name__ == "__main__":
    main()
