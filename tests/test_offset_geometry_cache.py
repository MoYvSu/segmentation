import json

import cv2
import numpy as np

from data.offset_geometry_dataset import OffsetGeometryDataset


def test_deterministic_geometry_sample_can_be_cached(tmp_path):
    image = np.full((20, 30, 3), 180, dtype=np.uint8)
    image_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(image_path), image)
    annotation = {
        "imageHeight": 20,
        "imageWidth": 30,
        "shapes": [
            {
                "label": "ferrite",
                "points": [[2, 2], [27, 2], [27, 17], [2, 17]],
                "shape_type": "polygon",
            }
        ],
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(annotation), encoding="utf-8"
    )
    dataset = OffsetGeometryDataset(
        tmp_path, image_size=32, output_grid=16, cache_in_memory=True
    )
    first = dataset[0]
    second = dataset[0]
    assert first["image"].data_ptr() == second["image"].data_ptr()
    assert first["offset_target"].data_ptr() == second["offset_target"].data_ptr()
    assert float(first["image"][:, -1].mean()) > 0.0
