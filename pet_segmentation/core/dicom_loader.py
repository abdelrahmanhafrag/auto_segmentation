"""
DICOM loading and preprocessing utilities for PET images.
"""

import numpy as np
import pydicom
import structlog
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import tempfile
import nibabel as nib

from ..config.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DICOMMetadata:
    """Container for DICOM metadata."""
    patient_id: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    series_description: Optional[str] = None
    pixel_spacing: Optional[Tuple[float, float]] = None
    slice_thickness: Optional[float] = None
    image_shape: Optional[Tuple[int, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, tuple):
                    result[key] = list(value)
                else:
                    result[key] = value
        return result


class DICOMLoader:
    """Enhanced DICOM loader with metadata extraction and validation."""
    
    def __init__(self, settings=None):
        """Initialize DICOM loader with settings."""
        self.settings = settings or get_settings()
        self.logger = get_logger(self.__class__.__name__)
    
    def load_dicom(self, dicom_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[DICOMMetadata]]:
        """
        Load DICOM file or series with comprehensive error handling.
        
        Args:
            dicom_path: Path to DICOM file or directory
            
        Returns:
            Tuple of (image_array, metadata) or (None, None) if failed
        """
        try:
            dicom_path = Path(dicom_path)
            
            if not dicom_path.exists():
                self.logger.error("DICOM path does not exist", path=str(dicom_path))
                return None, None
            
            # Single DICOM file
            if dicom_path.is_file():
                return self._load_single_dicom(dicom_path)
            
            # DICOM directory (series)
            elif dicom_path.is_dir():
                return self._load_dicom_series(dicom_path)
            
            else:
                self.logger.error("Invalid DICOM path", path=str(dicom_path))
                return None, None
                
        except Exception as e:
            self.logger.error("Failed to load DICOM", error=str(e), path=str(dicom_path))
            return None, None
    
    def _load_single_dicom(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[DICOMMetadata]]:
        """Load a single DICOM file."""
        try:
            self.logger.info("Loading single DICOM file", path=str(file_path))
            
            ds = pydicom.dcmread(str(file_path))
            
            # Validate modality for PET imaging
            if hasattr(ds, 'Modality') and ds.Modality != 'PT':
                self.logger.warning("Non-PET modality detected", modality=ds.Modality)
            
            # Extract image data
            image = ds.pixel_array
            
            # Handle different image orientations
            if hasattr(ds, 'ImageOrientationPatient'):
                # Could add orientation correction here if needed
                pass
            
            # Extract metadata
            metadata = self._extract_metadata(ds, image)
            
            self.logger.info(
                "Successfully loaded DICOM",
                shape=image.shape,
                dtype=str(image.dtype),
                modality=metadata.modality
            )
            
            return image, metadata
            
        except Exception as e:
            self.logger.error("Failed to load single DICOM", error=str(e), path=str(file_path))
            return None, None
    
    def _load_dicom_series(self, dir_path: Path) -> Tuple[Optional[np.ndarray], Optional[DICOMMetadata]]:
        """Load a DICOM series from directory."""
        try:
            self.logger.info("Loading DICOM series", path=str(dir_path))
            
            # Find DICOM files
            dicom_files = self._find_dicom_files(dir_path)
            
            if not dicom_files:
                self.logger.error("No DICOM files found in directory")
                return None, None
            
            self.logger.info("Found DICOM files", count=len(dicom_files))
            
            # Load and sort slices
            slices_data = []
            metadata = None
            
            for file_path in dicom_files:
                try:
                    ds = pydicom.dcmread(str(file_path))
                    
                    # Extract metadata from first valid slice
                    if metadata is None:
                        metadata = self._extract_metadata(ds, None)
                    
                    # Get slice position for sorting
                    slice_location = getattr(ds, 'SliceLocation', 0)
                    instance_number = getattr(ds, 'InstanceNumber', 0)
                    
                    slices_data.append({
                        'image': ds.pixel_array,
                        'slice_location': slice_location,
                        'instance_number': instance_number,
                        'file_path': file_path
                    })
                    
                except Exception as e:
                    self.logger.warning("Failed to load DICOM slice", file=str(file_path), error=str(e))
                    continue
            
            if not slices_data:
                self.logger.error("No valid DICOM slices found")
                return None, None
            
            # Sort slices by location or instance number
            slices_data.sort(key=lambda x: (x['slice_location'], x['instance_number']))
            
            # Stack into 3D volume
            images = [slice_data['image'] for slice_data in slices_data]
            volume = np.stack(images, axis=0)
            
            # Update metadata with volume info
            if metadata:
                metadata.image_shape = volume.shape
                metadata.min_value = float(volume.min())
                metadata.max_value = float(volume.max())
            
            self.logger.info(
                "Successfully loaded DICOM series",
                shape=volume.shape,
                dtype=str(volume.dtype),
                num_slices=len(slices_data)
            )
            
            return volume, metadata
            
        except Exception as e:
            self.logger.error("Failed to load DICOM series", error=str(e), path=str(dir_path))
            return None, None
    
    def _find_dicom_files(self, dir_path: Path) -> List[Path]:
        """Find DICOM files in directory."""
        extensions = ['.dcm', '.dicom', '']  # Include files without extension
        dicom_files = []
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                # Check by extension
                if any(str(file_path).lower().endswith(ext) for ext in extensions):
                    dicom_files.append(file_path)
                else:
                    # Try to read as DICOM (for files without extension)
                    try:
                        pydicom.dcmread(str(file_path), stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        continue
        
        return sorted(dicom_files)
    
    def _extract_metadata(self, ds: pydicom.Dataset, image: Optional[np.ndarray]) -> DICOMMetadata:
        """Extract relevant metadata from DICOM dataset."""
        metadata = DICOMMetadata()
        
        # Patient information
        metadata.patient_id = getattr(ds, 'PatientID', None)
        metadata.study_date = getattr(ds, 'StudyDate', None)
        
        # Image information
        metadata.modality = getattr(ds, 'Modality', None)
        metadata.series_description = getattr(ds, 'SeriesDescription', None)
        
        # Spatial information
        if hasattr(ds, 'PixelSpacing'):
            try:
                metadata.pixel_spacing = tuple(float(x) for x in ds.PixelSpacing)
            except:
                pass
        
        metadata.slice_thickness = getattr(ds, 'SliceThickness', None)
        if metadata.slice_thickness:
            metadata.slice_thickness = float(metadata.slice_thickness)
        
        # Image data information
        if image is not None:
            metadata.image_shape = image.shape
            metadata.min_value = float(image.min())
            metadata.max_value = float(image.max())
        
        return metadata
    
    def save_as_nifti(self, image: np.ndarray, output_path: Union[str, Path], 
                     metadata: Optional[DICOMMetadata] = None) -> bool:
        """
        Save image array as NIfTI format.
        
        Args:
            image: Image array to save
            output_path: Output file path
            metadata: Optional metadata for header information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            
            # Create affine matrix (identity if no spatial info)
            affine = np.eye(4)
            
            if metadata and metadata.pixel_spacing:
                # Set voxel spacing in affine matrix
                affine[0, 0] = metadata.pixel_spacing[0]
                affine[1, 1] = metadata.pixel_spacing[1]
                if metadata.slice_thickness:
                    affine[2, 2] = metadata.slice_thickness
            
            # Create NIfTI image
            nii_img = nib.Nifti1Image(image.astype(np.int16), affine)
            
            # Add metadata to header if available
            if metadata:
                header = nii_img.header
                if metadata.modality:
                    header.set_data_dtype(np.int16)
            
            # Save file
            nib.save(nii_img, str(output_path))
            
            self.logger.info("Saved NIfTI file", path=str(output_path), shape=image.shape)
            return True
            
        except Exception as e:
            self.logger.error("Failed to save NIfTI", error=str(e), path=str(output_path))
            return False
    
    def validate_pet_image(self, image: np.ndarray, metadata: Optional[DICOMMetadata] = None) -> Dict:
        """
        Validate that the loaded image is suitable for PET segmentation.
        
        Args:
            image: Loaded image array
            metadata: Image metadata
            
        Returns:
            Validation results dictionary
        """
        issues = []
        warnings = []
        
        # Check image dimensions
        if len(image.shape) not in [2, 3]:
            issues.append("Image must be 2D or 3D")
        
        # Check image size
        if image.size == 0:
            issues.append("Empty image")
        elif image.size > self.settings.max_file_size / 8:  # Rough size estimation
            warnings.append("Very large image - processing may be slow")
        
        # Check data type
        if not np.issubdtype(image.dtype, np.number):
            issues.append("Image must contain numeric data")
        
        # Check for PET-specific characteristics
        if metadata:
            if metadata.modality and metadata.modality != 'PT':
                warnings.append(f"Non-PET modality detected: {metadata.modality}")
            
            # Check for reasonable PET intensity range
            if metadata.min_value is not None and metadata.max_value is not None:
                if metadata.max_value <= metadata.min_value:
                    issues.append("Invalid intensity range")
                elif metadata.max_value < 100:  # Typical PET values are higher
                    warnings.append("Low maximum intensity - may not be SUV normalized")
        
        # Check for sufficient contrast
        if len(np.unique(image)) < 10:
            warnings.append("Low image contrast detected")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "metadata": metadata.to_dict() if metadata else None
        } 