"""
Main PET Segmentation orchestrator that combines all components.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time

from ..config.settings import get_settings
from ..utils.logging import get_logger, LoggerMixin
from .dicom_loader import DICOMLoader, DICOMMetadata
from .region_detector import RegionDetector
from .nninteractive_client import NNInteractiveClient
from .validator import SegmentationValidator


class PETSegmenter(LoggerMixin):
    """
    Main PET segmentation orchestrator that combines all components.
    
    This class provides a high-level interface for automated PET image segmentation
    using nnInteractive, with comprehensive error handling and quality control.
    """
    
    def __init__(self, nninteractive_url: Optional[str] = None, settings=None):
        """
        Initialize the PET segmenter.
        
        Args:
            nninteractive_url: URL of the nnInteractive server
            settings: Configuration settings object
        """
        self.settings = settings or get_settings()
        
        # Override URL if provided
        if nninteractive_url:
            self.settings.nninteractive_url = nninteractive_url
        
        # Initialize components
        self.dicom_loader = DICOMLoader(self.settings)
        self.region_detector = RegionDetector(self.settings)
        self.nn_client = NNInteractiveClient(self.settings)
        self.validator = SegmentationValidator(self.settings)
        
        self.logger.info(
            "PET Segmenter initialized",
            server_url=self.settings.nninteractive_url,
            max_regions=self.settings.max_regions_per_image
        )
    
    def test_server_connection(self) -> bool:
        """
        Test connection to nnInteractive server.
        
        Returns:
            True if connection successful, False otherwise
        """
        return self.nn_client.test_connection()
    
    def auto_segment_all_regions(
        self,
        dicom_path: Union[str, Path],
        max_regions: Optional[int] = None,
        quality_threshold: Optional[float] = None
    ) -> Dict:
        """
        Main function: automatically segment all bright regions from DICOM.
        
        Args:
            dicom_path: Path to DICOM file or directory
            max_regions: Maximum number of regions to segment
            quality_threshold: Minimum quality score to accept segmentation
            
        Returns:
            Dictionary with all segmentation results
        """
        start_time = time.time()
        
        # Use settings defaults if not specified
        max_regions = max_regions or self.settings.max_regions_per_image
        quality_threshold = quality_threshold or self.settings.min_quality_score
        
        self.logger.info(
            "Starting auto-segmentation",
            dicom_path=str(dicom_path),
            max_regions=max_regions,
            quality_threshold=quality_threshold
        )
        
        try:
            # Step 1: Load DICOM
            self.logger.info("Loading DICOM data")
            image, metadata = self.dicom_loader.load_dicom(dicom_path)
            
            if image is None:
                return {"error": "Failed to load DICOM data", "success": False}
            
            # Validate image
            validation_result = self.dicom_loader.validate_pet_image(image, metadata)
            if not validation_result["is_valid"]:
                return {
                    "error": f"Invalid PET image: {', '.join(validation_result['issues'])}",
                    "success": False
                }
            
            self.logger.info(
                "DICOM loaded successfully",
                shape=image.shape,
                modality=metadata.modality if metadata else "unknown"
            )
            
            # Step 2: Detect bright regions
            self.logger.info("Detecting bright regions")
            bright_regions = self.region_detector.detect_bright_regions(
                image,
                percentile_threshold=self.settings.default_percentile_threshold,
                min_size=self.settings.min_region_size
            )
            
            if not bright_regions:
                return {
                    "error": "No bright regions detected in image",
                    "success": False
                }
            
            self.logger.info(
                "Bright regions detected",
                total_regions=len(bright_regions)
            )
            
            # Limit to max_regions
            regions_to_process = bright_regions[:max_regions]
            
            # Step 3: Upload image to server
            self.logger.info("Uploading image to nnInteractive server")
            upload_success = self.nn_client.upload_image(image)
            
            if not upload_success:
                return {
                    "error": "Failed to upload image to nnInteractive server",
                    "success": False
                }
            
            # Step 4: Create new segment (optional)
            segment_id = self.nn_client.create_segment()
            
            # Step 5: Process each region
            all_segmentations = []
            successful_count = 0
            
            for i, (x, y, z) in enumerate(regions_to_process):
                self.logger.info(
                    "Processing region",
                    region_id=i+1,
                    center_point=(x, y, z),
                    total_regions=len(regions_to_process)
                )
                
                # Send point prompt
                segmentation_result = self.nn_client.send_point_prompt(x, y, z)
                
                if segmentation_result and "segmentation" in segmentation_result:
                    # Convert to numpy array if needed
                    segmentation = np.array(segmentation_result["segmentation"])
                    
                    # Validate quality
                    validation = self.validator.validate_segmentation(segmentation)
                    
                    # Check quality threshold
                    is_acceptable = validation["quality_score"] >= quality_threshold
                    
                    region_result = {
                        "region_id": i + 1,
                        "center_point": (x, y, z),
                        "segmentation": segmentation,
                        "validation": validation,
                        "brightness": float(image[z, y, x]),
                        "acceptable": is_acceptable
                    }
                    
                    all_segmentations.append(region_result)
                    
                    if is_acceptable:
                        successful_count += 1
                    
                    self.logger.info(
                        "Region segmentation completed",
                        region_id=i+1,
                        quality_score=validation["quality_score"],
                        volume=validation["volume"],
                        acceptable=is_acceptable
                    )
                    
                else:
                    self.logger.warning(
                        "Region segmentation failed",
                        region_id=i+1,
                        center_point=(x, y, z)
                    )
                    
                    all_segmentations.append({
                        "region_id": i + 1,
                        "center_point": (x, y, z),
                        "error": "Segmentation failed",
                        "acceptable": False
                    })
            
            # Compile final results
            processing_time = time.time() - start_time
            
            results = {
                "success": True,
                "total_regions_detected": len(bright_regions),
                "regions_processed": len(regions_to_process),
                "successful_segmentations": successful_count,
                "segmentations": all_segmentations,
                "processing_time": processing_time,
                "metadata": {
                    "image_shape": list(image.shape),
                    "dicom_metadata": metadata.to_dict() if metadata else None,
                    "quality_threshold": quality_threshold,
                    "settings": {
                        "percentile_threshold": self.settings.default_percentile_threshold,
                        "min_region_size": self.settings.min_region_size,
                        "max_regions": max_regions
                    }
                }
            }
            
            self.logger.info(
                "Auto-segmentation completed",
                processing_time=f"{processing_time:.2f}s",
                total_regions=len(bright_regions),
                successful_segmentations=successful_count
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Auto-segmentation failed",
                error=str(e),
                dicom_path=str(dicom_path)
            )
            
            return {
                "error": f"Segmentation failed: {str(e)}",
                "success": False,
                "processing_time": time.time() - start_time
            }
    
    def segment_single_region(
        self,
        image: np.ndarray,
        center_point: Tuple[int, int, int],
        use_bounding_box: bool = False,
        bbox_size: Optional[Tuple[int, int, int]] = None
    ) -> Optional[Dict]:
        """
        Segment a single region with specified center point.
        
        Args:
            image: 3D image array
            center_point: (x, y, z) coordinates of region center
            use_bounding_box: Whether to use bounding box prompt instead of point
            bbox_size: Size of bounding box if used
            
        Returns:
            Segmentation result dictionary or None if failed
        """
        try:
            # Upload image if not already uploaded
            if not self.nn_client.upload_image(image):
                return None
            
            x, y, z = center_point
            
            if use_bounding_box and bbox_size:
                # Calculate bounding box coordinates
                dx, dy, dz = bbox_size
                min_coords = (max(0, x - dx//2), max(0, y - dy//2), max(0, z - dz//2))
                max_coords = (
                    min(image.shape[2] - 1, x + dx//2),
                    min(image.shape[1] - 1, y + dy//2),
                    min(image.shape[0] - 1, z + dz//2)
                )
                
                result = self.nn_client.send_bounding_box_prompt(min_coords, max_coords)
            else:
                result = self.nn_client.send_point_prompt(x, y, z)
            
            if result and "segmentation" in result:
                segmentation = np.array(result["segmentation"])
                validation = self.validator.validate_segmentation(segmentation)
                
                return {
                    "segmentation": segmentation,
                    "validation": validation,
                    "center_point": center_point,
                    "method": "bounding_box" if use_bounding_box else "point"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(
                "Single region segmentation failed",
                error=str(e),
                center_point=center_point
            )
            return None
    
    def save_segmentation(
        self,
        segmentation: np.ndarray,
        output_path: Union[str, Path],
        metadata: Optional[DICOMMetadata] = None
    ) -> bool:
        """
        Save segmentation result to file.
        
        Args:
            segmentation: Segmentation array
            output_path: Output file path
            metadata: Optional DICOM metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            
            # Determine format from extension
            if output_path.suffix.lower() in ['.nii', '.gz']:
                return self.dicom_loader.save_as_nifti(segmentation, output_path, metadata)
            
            elif output_path.suffix.lower() == '.npy':
                np.save(output_path, segmentation)
                self.logger.info("Segmentation saved as NumPy", path=str(output_path))
                return True
            
            else:
                # Default to NIfTI
                nifti_path = output_path.with_suffix('.nii.gz')
                return self.dicom_loader.save_as_nifti(segmentation, nifti_path, metadata)
            
        except Exception as e:
            self.logger.error(
                "Failed to save segmentation",
                error=str(e),
                output_path=str(output_path)
            )
            return False
    
    def batch_process(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        max_regions_per_image: Optional[int] = None,
        parallel: bool = True
    ) -> Dict:
        """
        Process multiple DICOM files in batch.
        
        Args:
            input_paths: List of DICOM file/directory paths
            output_dir: Output directory for results
            max_regions_per_image: Max regions per image
            parallel: Whether to process in parallel
            
        Returns:
            Batch processing results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        max_regions = max_regions_per_image or self.settings.max_regions_per_image
        
        self.logger.info(
            "Starting batch processing",
            num_files=len(input_paths),
            output_dir=str(output_dir),
            parallel=parallel
        )
        
        results = []
        
        if parallel and len(input_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
                future_to_path = {
                    executor.submit(
                        self.auto_segment_all_regions,
                        path,
                        max_regions
                    ): path for path in input_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        result["input_path"] = str(path)
                        results.append(result)
                        
                        # Save individual results
                        filename = Path(path).stem
                        result_file = output_dir / f"{filename}_results.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                            
                    except Exception as e:
                        self.logger.error(
                            "Batch processing item failed",
                            path=str(path),
                            error=str(e)
                        )
                        results.append({
                            "input_path": str(path),
                            "error": str(e),
                            "success": False
                        })
        else:
            # Sequential processing
            for path in input_paths:
                try:
                    result = self.auto_segment_all_regions(path, max_regions)
                    result["input_path"] = str(path)
                    results.append(result)
                    
                    # Save individual results
                    filename = Path(path).stem
                    result_file = output_dir / f"{filename}_results.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                        
                except Exception as e:
                    self.logger.error(
                        "Batch processing item failed",
                        path=str(path),
                        error=str(e)
                    )
                    results.append({
                        "input_path": str(path),
                        "error": str(e),
                        "success": False
                    })
        
        # Compile batch summary
        successful_results = [r for r in results if r.get("success", False)]
        
        batch_summary = {
            "batch_success": True,
            "total_files": len(input_paths),
            "successful_files": len(successful_results),
            "failed_files": len(input_paths) - len(successful_results),
            "total_regions_found": sum(r.get("total_regions_detected", 0) for r in successful_results),
            "total_successful_segmentations": sum(r.get("successful_segmentations", 0) for r in successful_results),
            "results": results
        }
        
        # Save batch summary
        summary_file = output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        self.logger.info(
            "Batch processing completed",
            total_files=len(input_paths),
            successful_files=len(successful_results),
            output_dir=str(output_dir)
        )
        
        return batch_summary 