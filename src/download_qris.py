"""Download the pinned public QRIS CSV from its Mendeley Data DOI."""
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C

DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/xpkw64fd2h/files/"
    "f37b6b92-673b-4bb5-a635-c58b9b7051ba/file_downloaded"
)
EXPECTED_SHA256 = "1d31d8f918a1913ddc359957f0d369def124c6c9086b052398dd3a7032f014ee"


def run(output_path=None):
    output_path = Path(output_path or C.QRIS_FILE)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing QRIS file: {output_path}")
    request = Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "upi-consumer-trust-analysis/1.0"},
    )
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    if "text/csv" not in content_type.lower():
        raise ValueError(f"Expected a CSV response, received {content_type!r}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            "Downloaded QRIS file checksum does not match the DOI version: "
            f"{digest}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(payload)
    print(
        f"Downloaded DOI 10.17632/xpkw64fd2h.1 to {output_path} "
        f"({len(payload)} bytes; SHA-256 {digest})"
    )
    return output_path


if __name__ == "__main__":
    run()
