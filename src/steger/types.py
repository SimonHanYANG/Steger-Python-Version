"""Public data structures. Coordinates are zero-based (row, column), in pixels."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Contour:
    points: FloatArray
    normals: FloatArray
    response: FloatArray
    attributes: dict[str, NDArray] = field(default_factory=dict)
    closed: bool = False

    @property
    def rows(self) -> FloatArray:
        return self.points[:, 0]

    @property
    def cols(self) -> FloatArray:
        return self.points[:, 1]

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def get_attribute(self, name: str) -> NDArray:
        """HALCON angle attribute: radians CCW from the positive row axis."""
        if name == "angle":
            return np.mod(np.arctan2(self.normals[:, 1], self.normals[:, 0]), 2 * np.pi)
        if name == "response":
            return self.response
        return self.attributes[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.rows.tolist(),
            "col": self.cols.tolist(),
            "normal_row": self.normals[:, 0].tolist(),
            "normal_col": self.normals[:, 1].tolist(),
            "angle": self.get_attribute("angle").tolist(),
            "response": self.response.tolist(),
            "closed": self.closed,
            "attributes": {k: v.tolist() for k, v in self.attributes.items()},
        }


@dataclass
class Junction:
    point: FloatArray
    contour_indices: tuple[int, ...]
    completed: bool = False


@dataclass
class DetectionResult:
    contours: list[Contour]
    junctions: list[Junction]
    parameters: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_convention": "zero-based row,col; pixel centers at integers",
            "angle_convention": "normal pointing right of traversal; radians CCW from +row",
            "parameters": self.parameters,
            "diagnostics": self.diagnostics,
            "contours": [c.to_dict() for c in self.contours],
            "junctions": [
                {
                    "row": float(j.point[0]),
                    "col": float(j.point[1]),
                    "contours": list(j.contour_indices),
                    "completed": j.completed,
                }
                for j in self.junctions
            ],
        }
