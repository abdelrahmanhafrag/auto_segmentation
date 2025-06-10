"""
Segmentation validation and quality assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from dataclasses import dataclass

from ..config.settings import get_settings
from ..utils.logging import LoggerMixin


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    volume: int
    num_components: int
    largest_component_ratio: float
    surface_area: float
    compactness: float
    extent: Tuple[int, int, int, int, int, int]  # min_x, max_x, min_y, max_y, min_z, max_z
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "volume": self.volume,
            "num_components": self.num_components,
            "largest_component_ratio": self.largest_component_ratio,
            "surface_area": self.surface_area,
            "compactness": self.compactness,
            "extent": list(self.extent)
        }


class SegmentationValidator(LoggerMixin):
    """
    Validates segmentation quality using multiple criteria.
    """
    
    def __init__(self, settings=None):
        """Initialize validator with settings."""
        self.settings = settings or get_settings()
        
    def validate_segmentation(self, segmentation: np.ndarray) -> Dict:
        """
        Comprehensive validation of segmentation quality.
        
        Args:
            segmentation: Binary segmentation array
            
        Returns:
            Dictionary with validation results and quality score
        """
        try:
            self.logger.debug(
                "Validating segmentation",
                shape=segmentation.shape,
                unique_values=len(np.unique(segmentation))
            )
            
            # Ensure binary segmentation
            binary_seg = segmentation > 0
            
            if not np.any(binary_seg):
                return {
                    "quality_score": 0.0,
                    "issues": ["Empty segmentation"],
                    "warnings": [],
                    "is_acceptable": False,
                    "metrics": None
                }
            
            # Calculate metrics
            metrics = self._calculate_metrics(binary_seg)
            
            # Perform quality assessment
            quality_result = self._assess_quality(metrics)
            
            # Combine results
            result = {
                "quality_score": quality_result["score"],
                "issues": quality_result["issues"],
                "warnings": quality_result["warnings"],
                "is_acceptable": quality_result["score"] >= self.settings.min_quality_score,
                "metrics": metrics.to_dict(),
                "volume": metrics.volume,
                "num_components": metrics.num_components
            }
            
            self.logger.debug(
                "Validation completed",
                quality_score=result["quality_score"],
                volume=metrics.volume,
                components=metrics.num_components
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Segmentation validation failed", error=str(e))
            return {
                "quality_score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "is_acceptable": False,
                "metrics": None
            }
    
    def _calculate_metrics(self, binary_seg: np.ndarray) -> ValidationMetrics:
        """
        Calculate comprehensive metrics for segmentation.
        
        Args:
            binary_seg: Binary segmentation array
            
        Returns:
            ValidationMetrics object
        """
        # Basic volume
        volume = int(np.sum(binary_seg))
        
        # Connected components analysis
        labeled, num_components = ndimage.label(binary_seg)
        
        # Largest component ratio
        if num_components > 0:
            component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
            largest_component_size = max(component_sizes)
            largest_component_ratio = largest_component_size / volume
        else:
            largest_component_ratio = 0.0
        
        # Surface area estimation (using simple gradient method)
        surface_area = self._estimate_surface_area(binary_seg)
        
        # Compactness (sphere-like measure)
        compactness = self._calculate_compactness(volume, surface_area)
        
        # Extent (bounding box)
        extent = self._calculate_extent(binary_seg)
        
        return ValidationMetrics(
            volume=volume,
            num_components=num_components,
            largest_component_ratio=largest_component_ratio,
            surface_area=surface_area,
            compactness=compactness,
            extent=extent
        )
    
    def _estimate_surface_area(self, binary_seg: np.ndarray) -> float:
        """
        Estimate surface area using gradient-based method.
        
        Args:
            binary_seg: Binary segmentation
            
        Returns:
            Estimated surface area
        """
        try:
            # Use gradient to find boundaries
            grad_x = np.abs(np.gradient(binary_seg.astype(float), axis=2))
            grad_y = np.abs(np.gradient(binary_seg.astype(float), axis=1))
            grad_z = np.abs(np.gradient(binary_seg.astype(float), axis=0))
            
            # Surface area is where gradient is non-zero
            surface_area = np.sum(grad_x > 0) + np.sum(grad_y > 0) + np.sum(grad_z > 0)
            
            return float(surface_area)
            
        except Exception:
            # Fallback: simple boundary detection
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(binary_seg)
            boundary = binary_seg & ~eroded
            return float(np.sum(boundary))
    
    def _calculate_compactness(self, volume: int, surface_area: float) -> float:
        """
        Calculate compactness measure (sphere-like = 1.0).
        
        Args:
            volume: Segmentation volume
            surface_area: Estimated surface area
            
        Returns:
            Compactness measure
        """
        if surface_area <= 0 or volume <= 0:
            return 0.0
        
        # Theoretical surface area of sphere with same volume
        sphere_radius = (3 * volume / (4 * np.pi)) ** (1/3)
        sphere_surface_area = 4 * np.pi * sphere_radius ** 2
        
        # Compactness: closer to 1.0 means more sphere-like
        compactness = min(1.0, sphere_surface_area / surface_area)
        
        return compactness
    
    def _calculate_extent(self, binary_seg: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate bounding box extent.
        
        Args:
            binary_seg: Binary segmentation
            
        Returns:
            Tuple of (min_x, max_x, min_y, max_y, min_z, max_z)
        """
        coords = np.where(binary_seg)
        
        if len(coords[0]) == 0:
            return (0, 0, 0, 0, 0, 0)
        
        min_z, max_z = int(coords[0].min()), int(coords[0].max())
        min_y, max_y = int(coords[1].min()), int(coords[1].max())
        min_x, max_x = int(coords[2].min()), int(coords[2].max())
        
        return (min_x, max_x, min_y, max_y, min_z, max_z)
    
    def _assess_quality(self, metrics: ValidationMetrics) -> Dict:
        """
        Assess overall quality based on metrics.
        
        Args:
            metrics: Calculated metrics
            
        Returns:
            Quality assessment dictionary
        """
        score = 0.0
        issues = []
        warnings = []
        
        # Volume assessment (30% of score)
        volume_score = self._assess_volume(metrics.volume, issues, warnings)
        score += 0.3 * volume_score
        
        # Connectivity assessment (25% of score)
        connectivity_score = self._assess_connectivity(
            metrics.num_components,
            metrics.largest_component_ratio,
            issues,
            warnings
        )
        score += 0.25 * connectivity_score
        
        # Shape quality assessment (25% of score)
        shape_score = self._assess_shape_quality(metrics.compactness, issues, warnings)
        score += 0.25 * shape_score
        
        # Extent reasonableness (20% of score)
        extent_score = self._assess_extent(metrics.extent, issues, warnings)
        score += 0.2 * extent_score
        
        return {
            "score": min(1.0, score),
            "issues": issues,
            "warnings": warnings
        }
    
    def _assess_volume(self, volume: int, issues: List[str], warnings: List[str]) -> float:
        """Assess volume reasonableness."""
        if volume <= 0:
            issues.append("Empty segmentation")
            return 0.0
        
        if volume < 50:
            warnings.append("Very small segmentation volume")
            return 0.3
        
        if volume > 100000:
            warnings.append("Very large segmentation volume")
            return 0.8
        
        # Reasonable volume range
        if 100 <= volume <= 50000:
            return 1.0
        else:
            return 0.7
    
    def _assess_connectivity(
        self,
        num_components: int,
        largest_ratio: float,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Assess connectivity characteristics."""
        if num_components == 0:
            issues.append("No connected components found")
            return 0.0
        
        if num_components == 1:
            return 1.0  # Perfect - single connected component
        
        if num_components <= self.settings.max_components:
            if largest_ratio > 0.8:
                warnings.append(f"Multiple components ({num_components}) with dominant main component")
                return 0.7
            else:
                warnings.append(f"Multiple components ({num_components}) with fragmented structure")
                return 0.5
        else:
            issues.append(f"Too many disconnected components: {num_components}")
            return 0.2
    
    def _assess_shape_quality(self, compactness: float, issues: List[str], warnings: List[str]) -> float:
        """Assess shape quality based on compactness."""
        if compactness <= 0:
            warnings.append("Could not calculate shape compactness")
            return 0.5
        
        if compactness > 0.7:
            return 1.0  # Good compact shape
        elif compactness > 0.4:
            warnings.append("Moderately elongated or irregular shape")
            return 0.7
        else:
            warnings.append("Highly irregular or elongated shape")
            return 0.4
    
    def _assess_extent(self, extent: Tuple[int, int, int, int, int, int], issues: List[str], warnings: List[str]) -> float:
        """Assess spatial extent reasonableness."""
        min_x, max_x, min_y, max_y, min_z, max_z = extent
        
        # Calculate extent in each dimension
        x_extent = max_x - min_x + 1
        y_extent = max_y - min_y + 1
        z_extent = max_z - min_z + 1
        
        # Check for reasonable proportions
        max_extent = max(x_extent, y_extent, z_extent)
        min_extent = min(x_extent, y_extent, z_extent)
        
        if max_extent == 0:
            issues.append("Zero spatial extent")
            return 0.0
        
        aspect_ratio = max_extent / min_extent
        
        if aspect_ratio > 20:
            warnings.append("Highly elongated segmentation")
            return 0.4
        elif aspect_ratio > 10:
            warnings.append("Elongated segmentation")
            return 0.7
        else:
            return 1.0
    
    def compare_segmentations(
        self,
        seg1: np.ndarray,
        seg2: np.ndarray
    ) -> Dict:
        """
        Compare two segmentations using overlap metrics.
        
        Args:
            seg1: First segmentation
            seg2: Second segmentation
            
        Returns:
            Comparison metrics dictionary
        """
        try:
            # Ensure binary
            bin1 = seg1 > 0
            bin2 = seg2 > 0
            
            # Calculate overlap metrics
            intersection = np.sum(bin1 & bin2)
            union = np.sum(bin1 | bin2)
            vol1 = np.sum(bin1)
            vol2 = np.sum(bin2)
            
            # Dice coefficient
            dice = (2.0 * intersection) / (vol1 + vol2) if (vol1 + vol2) > 0 else 0.0
            
            # Jaccard index
            jaccard = intersection / union if union > 0 else 0.0
            
            # Volume similarity
            volume_similarity = 1.0 - abs(vol1 - vol2) / max(vol1, vol2, 1)
            
            return {
                "dice_coefficient": dice,
                "jaccard_index": jaccard,
                "volume_similarity": volume_similarity,
                "intersection_volume": intersection,
                "union_volume": union,
                "volume_1": vol1,
                "volume_2": vol2
            }
            
        except Exception as e:
            self.logger.error("Segmentation comparison failed", error=str(e))
            return {
                "dice_coefficient": 0.0,
                "jaccard_index": 0.0,
                "volume_similarity": 0.0,
                "error": str(e)
            } 