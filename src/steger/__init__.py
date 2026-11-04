"""Precision-oriented Steger line extraction. No HALCON installation required."""

from .detector import calculate_lines_gauss_parameters, lines_gauss
from .measurement import BendingMeasurement, measure_bending
from .types import Contour, DetectionResult, Junction

__version__ = "0.1.0"
__all__ = [
    "lines_gauss",
    "calculate_lines_gauss_parameters",
    "Contour",
    "DetectionResult",
    "Junction",
    "measure_bending",
    "BendingMeasurement",
]
