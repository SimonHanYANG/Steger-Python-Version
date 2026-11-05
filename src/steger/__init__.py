"""Precision-oriented Steger line extraction. No HALCON installation required."""

from .detector import calculate_lines_gauss_parameters, lines_gauss
from .measurement import (
    BendingMeasurement, NeckBendingHistory, measure_bending,
    measure_direction_angles, measure_end_direction_bending,
    measure_cpp_bending, RobustBendingMeasurement,
)
from .neck import measure_neck
from .types import Contour, DetectionResult, DetectionTrace, Junction

__version__ = "1.0.2"
__all__ = [
    "lines_gauss",
    "calculate_lines_gauss_parameters",
    "Contour",
    "DetectionResult",
    "DetectionTrace",
    "Junction",
    "measure_neck",
    "measure_bending",
    "measure_cpp_bending",
    "RobustBendingMeasurement",
    "BendingMeasurement",
    "measure_direction_angles",
    "measure_end_direction_bending",
    "NeckBendingHistory",
]
