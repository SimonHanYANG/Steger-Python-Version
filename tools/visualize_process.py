"""Create recorded process reports for the existing synthetic examples.

For real data, use steger detect ... --process-dir OUTPUT instead.
"""

import argparse
import json
from pathlib import Path

from steger import lines_gauss
from steger.io import read_image, write_json
from steger.synthetic import straight_line
from steger.visualization import save_process_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/process")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    image = read_image(root / "reports/examples/circular-neck.npy")
    mask = read_image(root / "reports/examples/circular-neck-mask.png", mask=True)
    result = lines_gauss(
        image, 1.5, 0.4, 1.2, "light", True, "gaussian", False, mask=mask, return_trace=True
    )
    save_process_report(
        result,
        output / "circular-neck",
        label="Synthetic circular arc with known geometry - not a real sperm image",
    )
    write_json(output / "circular-neck/result.json", result.to_dict())
    report = json.loads((root / "reports/accuracy.json").read_text())
    case = next(
        c
        for c in report["cases"]
        if c["case"]["model"] == "gaussian"
        and c["case"].get("noise") == 3
        and c["case"].get("polarity") == "dark"
    )
    image, _ = straight_line(shape=(72, 72), **case["case"])
    result = lines_gauss(
        image, case["sigma"], 0.4, 1.2, "dark", True, "gaussian", False, return_trace=True
    )
    save_process_report(
        result,
        output / "noisy-line",
        label="Synthetic dark Gaussian line - noise SD 3, asymmetric backgrounds, 8-bit",
    )
    write_json(output / "noisy-line/result.json", result.to_dict())
    print(output / "circular-neck/README.md")
    print(output / "noisy-line/README.md")


if __name__ == "__main__":
    main()
