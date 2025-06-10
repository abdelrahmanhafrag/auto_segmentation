"""
PET Segmentation: Automated PET imaging segmentation using nnInteractive

A scalable solution for automated segmentation of PET (Positron Emission Tomography) 
images without requiring CT reference scans, leveraging the nnInteractive framework
for interactive medical image segmentation.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core.segmenter import PETSegmenter
from .core.dicom_loader import DICOMLoader
from .core.region_detector import RegionDetector
from .config.settings import Settings

__all__ = [
    "__version__",
    "PETSegmenter", 
    "DICOMLoader",
    "RegionDetector",
    "Settings",
] 