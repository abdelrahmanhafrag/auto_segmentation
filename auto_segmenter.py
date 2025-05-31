#!/usr/bin/env python3
"""
DICOM Auto-Segmentation Script
Takes a DICOM file and automatically segments organs using nnInteractive
"""

import requests
import numpy as np
import pydicom
import json
import time
from pathlib import Path
import nibabel as nib
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO

class DICOMAutoSegmenter:
    def __init__(self, nninteractive_url: str = "http://localhost:1527"):
        """
        Initialize the auto-segmentation system
        
        Args:
            nninteractive_url: URL of the nnInteractive server
        """
        self.server_url = nninteractive_url
        self.session_id = None
        
    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """
        Load DICOM file simply - just like 3D Slicer does
        
        Args:
            dicom_path: Path to DICOM file or directory
            
        Returns:
            3D numpy array of the medical image (original values)
        """
        try:
            # Single DICOM file
            if Path(dicom_path).is_file():
                ds = pydicom.dcmread(dicom_path)
                # Return original pixel values - no processing
                return ds.pixel_array
            
            # DICOM directory (DICOM series)
            elif Path(dicom_path).is_dir():
                dicom_files = list(Path(dicom_path).glob("*.dcm"))
                if not dicom_files:
                    raise ValueError("No DICOM files found in directory")
                
                # Just load and stack - like Slicer does
                slices = []
                for file in sorted(dicom_files):
                    ds = pydicom.dcmread(file)
                    slices.append(ds.pixel_array)
                
                # Stack into 3D volume
                return np.stack(slices, axis=0)
            
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            return None
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Optional light preprocessing - only if nnInteractive server needs it
        Most of the time, just return original image
        """
        # Just return original - let nnInteractive handle it
        return image
    
    def detect_all_bright_regions(self, image: np.ndarray, percentile_threshold: int = 90, 
                                 min_size: int = 100) -> List[Tuple[int, int, int]]:
        """
        Detect bright regions using simple percentile threshold - like visual inspection
        
        Args:
            image: 3D medical image (original DICOM values)
            percentile_threshold: Use pixels above this percentile as "bright" (90 = top 10%)
            min_size: Minimum size of region to consider
            
        Returns:
            List of (x, y, z) coordinates of bright region centers
        """
        from scipy import ndimage
        from skimage import measure
        
        # Find brightness threshold (top X% of pixel values)
        threshold_value = np.percentile(image, percentile_threshold)
        
        # Simple binary mask: bright vs not bright
        bright_mask = image > threshold_value
        
        # Basic cleanup - remove tiny spots
        bright_mask = ndimage.binary_opening(bright_mask, iterations=1)
        
        # Find connected regions
        labeled_regions = measure.label(bright_mask)
        regions = measure.regionprops(labeled_regions)
        
        bright_centers = []
        
        for region in regions:
            # Skip tiny regions
            if region.area < min_size:
                continue
                
            # Get center point
            center = region.centroid
            
            # Convert to coordinates (z, y, x) -> (x, y, z)
            x = int(center[2]) if len(center) > 2 else int(center[1])
            y = int(center[1]) if len(center) > 2 else int(center[0])
            z = int(center[0]) if len(center) > 2 else 0
            
            # Check bounds
            if (0 <= x < image.shape[-1] and 
                0 <= y < image.shape[-2] and 
                0 <= z < image.shape[0]):
                
                bright_centers.append((x, y, z))
        
        # Sort by actual pixel intensity at center
        bright_centers.sort(key=lambda coord: image[coord[2], coord[1], coord[0]], reverse=True)
        
        return bright_centers
    
    def upload_image_to_server(self, image: np.ndarray) -> str:
        """
        Upload image to nnInteractive server - keep it simple
        
        Returns:
            Session ID for further communication
        """
        try:
            # Send image data as-is, let server handle format
            # In reality, might need to save as temp file and upload that
            files = {'file': ('image.npy', image.tobytes())}
            
            response = requests.post(
                f"{self.server_url}/upload",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = result.get("session_id")
                return self.session_id
            else:
                print(f"Failed to upload image: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None
    
    def send_point_prompt(self, x: int, y: int, z: int, is_positive: bool = True) -> Dict:
        """
        Send point prompt to nnInteractive server
        
        Args:
            x, y, z: Coordinates of the point
            is_positive: Whether this is a positive or negative prompt
            
        Returns:
            Segmentation result
        """
        try:
            prompt_data = {
                "session_id": self.session_id,
                "prompt_type": "point",
                "coordinates": [x, y, z],
                "is_positive": is_positive,
                "slice_index": z
            }
            
            response = requests.post(
                f"{self.server_url}/segment",
                json=prompt_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Segmentation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error sending prompt: {e}")
            return None
    
    def send_bounding_box_prompt(self, min_coords: Tuple[int, int, int], 
                                max_coords: Tuple[int, int, int]) -> Dict:
        """
        Send bounding box prompt to nnInteractive server
        """
        try:
            prompt_data = {
                "session_id": self.session_id,
                "prompt_type": "bounding_box",
                "min_coordinates": list(min_coords),
                "max_coordinates": list(max_coords)
            }
            
            response = requests.post(
                f"{self.server_url}/segment",
                json=prompt_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Bounding box segmentation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error sending bounding box: {e}")
            return None
    
    def validate_segmentation(self, segmentation: np.ndarray) -> Dict:
        """
        Basic validation of segmentation quality - no organ-specific assumptions
        """
        try:
            # Calculate basic metrics
            volume = np.sum(segmentation > 0)
            
            # Get connected components
            labeled, num_components = ndimage.label(segmentation > 0)
            
            # Basic quality checks
            quality_score = 0.0
            issues = []
            
            # Volume check (general reasonable range)
            if 50 < volume < 100000:  # Reasonable general volume range
                quality_score += 0.4
            else:
                issues.append("Volume outside reasonable range")
            
            # Connectivity check - prefer single connected component
            if num_components == 1:
                quality_score += 0.4
            elif num_components <= 3:
                quality_score += 0.2
                issues.append(f"Multiple components: {num_components}")
            else:
                issues.append(f"Too many disconnected components: {num_components}")
            
            # Shape check
            if len(segmentation.shape) == 3:
                quality_score += 0.2
            
            return {
                "quality_score": quality_score,
                "volume": int(volume),
                "num_components": num_components,
                "issues": issues,
                "is_acceptable": quality_score > 0.5
            }
            
        except Exception as e:
            return {
                "quality_score": 0.0,
                "issues": [f"Validation error: {e}"],
                "is_acceptable": False
            }
    
    def auto_segment_all_regions(self, dicom_path: str, max_regions: int = 10) -> Dict:
        """
        Main function: automatically segment all bright regions from DICOM
        
        Args:
            dicom_path: Path to DICOM file or directory
            max_regions: Maximum number of regions to segment
            
        Returns:
            Dictionary with all segmentation results
        """
        print(f"🔄 Starting auto-segmentation of all bright regions from {dicom_path}")
        
        # Step 1: Load DICOM
        print("📁 Loading DICOM...")
        image = self.load_dicom(dicom_path)
        if image is None:
            return {"error": "Failed to load DICOM"}
        
        # Step 2: Skip preprocessing - use original DICOM data
        print("✅ Using original DICOM data (no preprocessing)")
        # image = self.normalize_image(image)  # Skip this step
        
        # Step 3: Detect all bright regions
        print("🔍 Detecting all bright regions...")
        bright_regions = self.detect_all_bright_regions(image)
        
        if not bright_regions:
            return {"error": "No bright regions detected in image"}
        
        print(f"✨ Found {len(bright_regions)} bright regions")
        
        # Limit to max_regions to avoid overwhelming the system
        regions_to_process = bright_regions[:max_regions]
        
        # Step 4: Upload to server
        print("⬆️ Uploading to nnInteractive server...")
        session_id = self.upload_image_to_server(image)
        if not session_id:
            return {"error": "Failed to upload image to server"}
        
        # Step 5: Segment each bright region
        all_segmentations = []
        
        for i, (x, y, z) in enumerate(regions_to_process):
            print(f"🎯 Segmenting region {i+1}/{len(regions_to_process)} at ({x}, {y}, {z})")
            
            # Send point prompt for this region
            result = self.send_point_prompt(x, y, z)
            
            if result and "segmentation" in result:
                # Convert result to numpy array
                segmentation = np.array(result["segmentation"])
                
                # Validate quality
                validation = self.validate_segmentation(segmentation)
                
                region_result = {
                    "region_id": i + 1,
                    "center_point": (x, y, z),
                    "segmentation": segmentation,
                    "validation": validation,
                    "brightness": float(image[z, y, x])  # Brightness at center point
                }
                
                all_segmentations.append(region_result)
                
                print(f"✅ Region {i+1} - Quality: {validation['quality_score']:.2f}, "
                      f"Volume: {validation['volume']}")
            else:
                print(f"❌ Failed to segment region {i+1}")
                all_segmentations.append({
                    "region_id": i + 1,
                    "center_point": (x, y, z),
                    "error": "Segmentation failed"
                })
        
        return {
            "success": True,
            "total_regions_detected": len(bright_regions),
            "regions_processed": len(regions_to_process),
            "segmentations": all_segmentations,
            "metadata": {
                "session_id": session_id,
                "image_shape": list(image.shape)
            }
        }
    
    def save_segmentation(self, segmentation: np.ndarray, output_path: str):
        """
        Save segmentation result to file
        """
        try:
            # Save as NIfTI format (common for medical segmentations)
            if output_path.endswith('.nii') or output_path.endswith('.nii.gz'):
                nii_img = nib.Nifti1Image(segmentation.astype(np.uint8), np.eye(4))
                nib.save(nii_img, output_path)
                print(f"💾 Segmentation saved to: {output_path}")
            
            # Save as numpy array
            elif output_path.endswith('.npy'):
                np.save(output_path, segmentation)
                print(f"💾 Segmentation saved to: {output_path}")
            
            else:
                # Default to numpy
                np.save(output_path + '.npy', segmentation)
                print(f"💾 Segmentation saved to: {output_path}.npy")
                
        except Exception as e:
            print(f"❌ Error saving segmentation: {e}")


def main():
    """
    Example usage of the DICOM auto-segmentation system - detects all bright regions
    """
    # Initialize the segmenter
    segmenter = DICOMAutoSegmenter(nninteractive_url="http://localhost:1527")
    
    # Path to your DICOM file or directory
    dicom_path = input("Enter path to DICOM file or directory: ").strip()
    
    # Max regions to process
    max_regions = int(input("Max regions to segment (default 5): ").strip() or "5")
    
    print(f"\n🚀 Starting automatic detection and segmentation...")
    print(f"📂 DICOM path: {dicom_path}")
    print(f"🎯 Max regions: {max_regions}")
    print(f"🌐 nnInteractive server: {segmenter.server_url}")
    print("-" * 50)
    
    # Run auto-segmentation for all bright regions
    result = segmenter.auto_segment_all_regions(dicom_path, max_regions)
    
    # Display results
    print("\n" + "="*50)
    print("📊 SEGMENTATION RESULTS")
    print("="*50)
    
    if result.get("success"):
        segmentations = result.get("segmentations", [])
        
        print(f"✅ Status: Success")
        print(f"🔍 Total regions detected: {result.get('total_regions_detected', 0)}")
        print(f"🎯 Regions processed: {result.get('regions_processed', 0)}")
        print(f"✅ Successful segmentations: {len([s for s in segmentations if 'segmentation' in s])}")
        
        # Show details for each segmentation
        for seg in segmentations:
            if 'segmentation' in seg:
                validation = seg.get('validation', {})
                print(f"\n🏷️ Region {seg['region_id']}:")
                print(f"   📍 Center: {seg['center_point']}")
                print(f"   💡 Brightness: {seg['brightness']:.3f}")
                print(f"   📊 Quality: {validation.get('quality_score', 0):.2f}")
                print(f"   📏 Volume: {validation.get('volume', 0)} voxels")
                print(f"   🔗 Components: {validation.get('num_components', 0)}")
                
                if validation.get("issues"):
                    print(f"   ⚠️ Issues: {', '.join(validation['issues'])}")
                
                # Save each segmentation
                output_path = f"region_{seg['region_id']}_segmentation_{int(time.time())}.nii.gz"
                segmenter.save_segmentation(seg["segmentation"], output_path)
        
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n🏁 Auto-segmentation complete!")


if __name__ == "__main__":
    main()