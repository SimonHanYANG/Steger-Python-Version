import json

import numpy as np
from PIL import Image

from steger.cli import main
from steger.io import read_image, write_json
from steger.synthetic import straight_line


def test_sixteen_bit_pixels_are_preserved(tmp_path):
    data = np.array([[0, 1024], [2048, 65535]], dtype=np.uint16)
    path = tmp_path / "image.tiff"
    Image.fromarray(data).save(path)
    assert np.array_equal(read_image(path), data)


def test_nonfinite_quality_attributes_are_standard_json_null(tmp_path):
    path = tmp_path / "quality.json"
    write_json(path, {"rms": [np.nan, np.inf, np.float64(0.1)]})
    assert json.loads(path.read_text()) == {"rms": [None, None, 0.1]}


def test_cli_round_trip_and_explicit_neck_measurement(tmp_path):
    image, _ = straight_line(shape=(48, 48), model="gaussian", angle_deg=31)
    source = tmp_path / "input.npy"
    np.save(source, image)
    output = tmp_path / "result.json"
    overlay = tmp_path / "overlay.png"
    status = main(
        [
            "detect",
            str(source),
            "--sigma",
            "1.5",
            "--low",
            ".5",
            "--high",
            "2",
            "--model",
            "gaussian",
            "--no-complete-junctions",
            "--measure-neck",
            "--output",
            str(output),
            "--overlay",
            str(overlay),
        ]
    )
    assert status == 0
    data = json.loads(output.read_text())
    assert data["neck_measurement"]["angle_deg"] < 0.01
    assert data["diagnostics"]["halcon_parity"] == "not_verified"
    assert len(data["image_sha256"]) == 64
    assert Image.open(overlay).size == (144, 144)
