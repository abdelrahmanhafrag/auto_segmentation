"""
Automated region detection for PET images.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage import measure

from ..config.settings import get_settings
from ..utils.logging import LoggerMixin


class RegionDetector(LoggerMixin):
    """
    Detects bright regions in PET images using intelligent thresholding
    and morphological operations.
    """
    
    def __init__(self, settings=None):
        """Initialize region detector with settings."""
        self.settings = settings or get_settings()
        
    def detect_bright_regions(
        self,
        image: np.ndarray,
        percentile_threshold: int = 90,
        min_size: int = 100,
        max_regions: Optional[int] = None
    ) -> List[Tuple[int, int, int]]:
        """
        Detect bright regions using percentile-based thresholding.
        
        Args:
            image: 3D medical image array
            percentile_threshold: Percentile threshold for brightness (50-99)
            min_size: Minimum region size in voxels
            max_regions: Maximum number of regions to return
            
        Returns:
            List of (x, y, z) coordinates of region centers
        """
        self.logger.info(
            "Detecting bright regions",
            image_shape=image.shape,
            percentile_threshold=percentile_threshold,
            min_size=min_size
        )
        
        try:
            # Calculate threshold value
            threshold_value = np.percentile(image, percentile_threshold)
            
            self.logger.debug(
                "Threshold calculated",
                percentile=percentile_threshold,
                threshold_value=threshold_value,
                image_min=image.min(),
                image_max=image.max()
            )
            
            # Create binary mask
            bright_mask = image > threshold_value
            
            # Morphological operations for cleanup
            bright_mask = self._cleanup_mask(bright_mask)
            
            # Find connected components
            regions = self._find_connected_regions(bright_mask, min_size)
            
            # Extract region centers
            centers = self._extract_region_centers(regions, image)
            
            # Limit number of regions if specified
            if max_regions and len(centers) > max_regions:
                centers = centers[:max_regions]
            
            self.logger.info(
                "Region detection completed",
                regions_found=len(centers),
                threshold_value=threshold_value
            )
            
            return centers
            
        except Exception as e:
            self.logger.error("Region detection failed", error=str(e))
            return []
    
    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up binary mask using morphological operations.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Cleaned binary mask
        """
        # Remove small holes and noise
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=1)
        
        # Fill small holes
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    def _find_connected_regions(
        self,
        mask: np.ndarray,
        min_size: int
    ) -> List[measure._regionprops.RegionProperties]:
        """
        Find connected regions in binary mask.
        
        Args:
            mask: Binary mask
            min_size: Minimum region size
            
        Returns:
            List of region properties
        """
        # Label connected components
        labeled_regions = measure.label(mask)
        
        # Extract region properties
        regions = measure.regionprops(labeled_regions)
        
        # Filter by size
        filtered_regions = [
            region for region in regions
            if region.area >= min_size
        ]
        
        self.logger.debug(
            "Connected regions found",
            total_regions=len(regions),
            filtered_regions=len(filtered_regions),
            min_size=min_size
        )
        
        return filtered_regions
    
    def _extract_region_centers(
        self,
        regions: List[measure._regionprops.RegionProperties],
        image: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """
        Extract center coordinates from regions.
        
        Args:
            regions: List of region properties
            image: Original image for intensity-based sorting
            
        Returns:
            List of (x, y, z) center coordinates sorted by intensity
        """
        centers = []
        
        for region in regions:
            centroid = region.centroid
            
            # Convert to (x, y, z) coordinates
            if len(centroid) == 3:
                z, y, x = centroid
            elif len(centroid) == 2:
                y, x = centroid
                z = 0
            else:
                continue
            
            # Ensure coordinates are within bounds
            x = max(0, min(int(x), image.shape[-1] - 1))
            y = max(0, min(int(y), image.shape[-2] - 1))
            z = max(0, min(int(z), image.shape[0] - 1))
            
            centers.append((x, y, z))
        
        # Sort by intensity at center point (brightest first)
        centers.sort(
            key=lambda coord: image[coord[2], coord[1], coord[0]],
            reverse=True
        )
        
        return centers
    
    def detect_regions_with_adaptive_threshold(
        self,
        image: np.ndarray,
        initial_percentile: int = 95,
        min_regions: int = 3,
        max_regions: int = 15
    ) -> List[Tuple[int, int, int]]:
        """
        Detect regions using adaptive thresholding to ensure optimal number.
        
        Args:
            image: 3D medical image
            initial_percentile: Starting percentile threshold
            min_regions: Minimum desired regions
            max_regions: Maximum desired regions
            
        Returns:
            List of region centers
        """
        self.logger.info(
            "Starting adaptive region detection",
            initial_percentile=initial_percentile,
            min_regions=min_regions,
            max_regions=max_regions
        )
        
        percentile = initial_percentile
        step = 2
        
        for attempt in range(10):  # Max 10 attempts
            regions = self.detect_bright_regions(
                image,
                percentile_threshold=percentile,
                min_size=self.settings.min_region_size
            )
            
            num_regions = len(regions)
            
            self.logger.debug(
                "Adaptive threshold attempt",
                attempt=attempt + 1,
                percentile=percentile,
                regions_found=num_regions
            )
            
            if min_regions <= num_regions <= max_regions:
                self.logger.info(
                    "Optimal region count achieved",
                    final_percentile=percentile,
                    regions_found=num_regions
                )
                return regions
            
            # Adjust threshold
            if num_regions < min_regions:
                # Too few regions, lower threshold
                percentile = max(50, percentile - step)
            else:
                # Too many regions, raise threshold
                percentile = min(99, percentile + step)
            
            # Increase step size for faster convergence
            step = min(5, step + 1)
        
        # Fallback: use last result
        self.logger.warning(
            "Could not achieve optimal region count",
            final_percentile=percentile,
            final_regions=len(regions)
        )
        
        return regions
    
    def analyze_region_characteristics(
        self,
        image: np.ndarray,
        regions: List[Tuple[int, int, int]]
    ) -> List[dict]:
        """
        Analyze characteristics of detected regions.
        
        Args:
            image: Original image
            regions: List of region centers
            
        Returns:
            List of region characteristic dictionaries
        """
        characteristics = []
        
        for i, (x, y, z) in enumerate(regions):
            # Extract local neighborhood
            neighborhood = self._extract_neighborhood(image, x, y, z)
            
            char = {
                "region_id": i + 1,
                "center": (x, y, z),
                "center_intensity": float(image[z, y, x]),
                "neighborhood_mean": float(np.mean(neighborhood)),
                "neighborhood_std": float(np.std(neighborhood)),
                "neighborhood_max": float(np.max(neighborhood)),
                "contrast": float(image[z, y, x] - np.mean(neighborhood)),
            }
            
            characteristics.append(char)
        
        return characteristics
    
    def _extract_neighborhood(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        z: int,
        radius: int = 5
    ) -> np.ndarray:
        """
        Extract neighborhood around a point.
        
        Args:
            image: 3D image
            x, y, z: Center coordinates
            radius: Neighborhood radius
            
        Returns:
            Neighborhood array
        """
        # Define bounds
        z_min = max(0, z - radius)
        z_max = min(image.shape[0], z + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(image.shape[1], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(image.shape[2], x + radius + 1)
        
        return image[z_min:z_max, y_min:y_max, x_min:x_max] 