"""
Client for communicating with nnInteractive server.
"""

import requests
import numpy as np
import tempfile
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import time

from ..config.settings import get_settings
from ..utils.logging import LoggerMixin


class NNInteractiveClient(LoggerMixin):
    """
    Client for communicating with nnInteractive server.
    Handles image upload, prompts, and segmentation requests.
    """
    
    def __init__(self, settings=None):
        """Initialize nnInteractive client."""
        self.settings = settings or get_settings()
        self.server_url = self.settings.nninteractive_url
        self.timeout = self.settings.nninteractive_timeout
        self.retries = self.settings.nninteractive_retries
        
        self.logger.info(
            "nnInteractive client initialized",
            server_url=self.server_url,
            timeout=self.timeout
        )
    
    def test_connection(self) -> bool:
        """
        Test connection to nnInteractive server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing server connection", url=self.server_url)
            
            # Try root endpoint
            response = requests.get(
                f"{self.server_url}/",
                timeout=10
            )
            
            self.logger.debug(
                "Root endpoint response",
                status_code=response.status_code
            )
            
            # Try docs endpoint for API documentation
            try:
                docs_response = requests.get(
                    f"{self.server_url}/docs",
                    timeout=10
                )
                self.logger.debug(
                    "Docs endpoint response",
                    status_code=docs_response.status_code
                )
            except:
                self.logger.debug("Docs endpoint not available")
            
            # Consider connection successful if we get any response
            self.logger.info("Server connection successful")
            return True
            
        except Exception as e:
            self.logger.error("Server connection failed", error=str(e))
            return False
    
    def upload_image(self, image: np.ndarray) -> bool:
        """
        Upload image to nnInteractive server.
        
        Args:
            image: 3D image array to upload
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            self.logger.info(
                "Uploading image to server",
                shape=image.shape,
                dtype=str(image.dtype)
            )
            
            # Retry mechanism
            for attempt in range(self.retries):
                try:
                    success = self._upload_attempt(image)
                    if success:
                        self.logger.info("Image upload successful")
                        return True
                    
                except Exception as e:
                    self.logger.warning(
                        "Upload attempt failed",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    
                    if attempt < self.retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            self.logger.error("All upload attempts failed")
            return False
            
        except Exception as e:
            self.logger.error("Image upload failed", error=str(e))
            return False
    
    def _upload_attempt(self, image: np.ndarray) -> bool:
        """
        Single upload attempt.
        
        Args:
            image: Image array to upload
            
        Returns:
            True if successful
        """
        # Create temporary NIfTI file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            try:
                # Save as NIfTI format (medical standard)
                nii_img = nib.Nifti1Image(image.astype(np.int16), np.eye(4))
                nib.save(nii_img, tmp_file.name)
                
                self.logger.debug(
                    "Created temporary NIfTI file",
                    path=tmp_file.name,
                    size_mb=Path(tmp_file.name).stat().st_size / (1024**2)
                )
                
                # Upload the file
                with open(tmp_file.name, 'rb') as f:
                    files = {
                        'file': ('medical_image.nii.gz', f, 'application/octet-stream')
                    }
                    
                    response = requests.post(
                        f"{self.server_url}/upload_image",
                        files=files,
                        timeout=self.timeout
                    )
                
                self.logger.debug(
                    "Upload response received",
                    status_code=response.status_code
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "ok":
                        return True
                    else:
                        self.logger.warning(
                            "Unexpected upload response",
                            response=result
                        )
                        return True  # Proceed anyway
                else:
                    self.logger.error(
                        "Upload failed",
                        status_code=response.status_code,
                        response_text=response.text[:500]
                    )
                    return False
                    
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_file.name).unlink()
                except:
                    pass
    
    def send_point_prompt(
        self,
        x: int,
        y: int,
        z: int,
        is_positive: bool = True
    ) -> Optional[Dict]:
        """
        Send point prompt to nnInteractive server.
        
        Args:
            x, y, z: Point coordinates
            is_positive: Whether this is a positive prompt
            
        Returns:
            Segmentation result or None if failed
        """
        try:
            prompt_data = {
                "voxel_coord": [int(x), int(y), int(z)],
                "positive_click": bool(is_positive)
            }
            
            self.logger.debug(
                "Sending point prompt",
                coordinates=(x, y, z),
                positive=is_positive
            )
            
            # Retry mechanism
            for attempt in range(self.retries):
                try:
                    response = requests.post(
                        f"{self.server_url}/add_point_interaction",
                        json=prompt_data,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        self.logger.debug(
                            "Point prompt successful",
                            response_keys=list(result.keys()) if isinstance(result, dict) else None
                        )
                        
                        return result
                    else:
                        self.logger.warning(
                            "Point prompt failed",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            response_text=response.text[:200]
                        )
                        
                        if attempt < self.retries - 1:
                            time.sleep(1)
                
                except requests.exceptions.RequestException as e:
                    self.logger.warning(
                        "Point prompt request failed",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    
                    if attempt < self.retries - 1:
                        time.sleep(2 ** attempt)
            
            self.logger.error("All point prompt attempts failed")
            return None
            
        except Exception as e:
            self.logger.error("Point prompt failed", error=str(e))
            return None
    
    def send_bounding_box_prompt(
        self,
        min_coords: Tuple[int, int, int],
        max_coords: Tuple[int, int, int],
        is_positive: bool = True
    ) -> Optional[Dict]:
        """
        Send bounding box prompt to nnInteractive server.
        
        Args:
            min_coords: Minimum coordinates (x, y, z)
            max_coords: Maximum coordinates (x, y, z)
            is_positive: Whether this is a positive prompt
            
        Returns:
            Segmentation result or None if failed
        """
        try:
            prompt_data = {
                "outer_point_one": [min_coords[0], min_coords[1], min_coords[2]],
                "outer_point_two": [max_coords[0], max_coords[1], max_coords[2]],
                "positive_click": bool(is_positive)
            }
            
            self.logger.debug(
                "Sending bounding box prompt",
                min_coords=min_coords,
                max_coords=max_coords,
                positive=is_positive
            )
            
            # Retry mechanism
            for attempt in range(self.retries):
                try:
                    response = requests.post(
                        f"{self.server_url}/add_bbox_interaction",
                        json=prompt_data,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        self.logger.debug(
                            "Bounding box prompt successful",
                            response_keys=list(result.keys()) if isinstance(result, dict) else None
                        )
                        
                        return result
                    else:
                        self.logger.warning(
                            "Bounding box prompt failed",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            response_text=response.text[:200]
                        )
                        
                        if attempt < self.retries - 1:
                            time.sleep(1)
                
                except requests.exceptions.RequestException as e:
                    self.logger.warning(
                        "Bounding box request failed",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    
                    if attempt < self.retries - 1:
                        time.sleep(2 ** attempt)
            
            self.logger.error("All bounding box prompt attempts failed")
            return None
            
        except Exception as e:
            self.logger.error("Bounding box prompt failed", error=str(e))
            return None
    
    def create_segment(self) -> str:
        """
        Create a new segment on the server.
        
        Returns:
            Segment ID or default value
        """
        try:
            self.logger.debug("Creating new segment")
            
            response = requests.post(
                f"{self.server_url}/upload_segment",
                json={},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                segment_id = result.get("segment_id") or result.get("id") or "default"
                
                self.logger.debug(
                    "Segment created successfully",
                    segment_id=segment_id
                )
                
                return segment_id
            else:
                self.logger.warning(
                    "Segment creation response",
                    status_code=response.status_code,
                    response_text=response.text[:200]
                )
                return "default"
                
        except Exception as e:
            self.logger.warning("Segment creation failed", error=str(e))
            return "default"
    
    def get_server_info(self) -> Optional[Dict]:
        """
        Get server information and capabilities.
        
        Returns:
            Server info dictionary or None
        """
        try:
            response = requests.get(
                f"{self.server_url}/info",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(
                    "Server info not available",
                    status_code=response.status_code
                )
                return None
                
        except Exception as e:
            self.logger.debug("Could not get server info", error=str(e))
            return None
    
    def reset_session(self) -> bool:
        """
        Reset the current session on the server.
        
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.server_url}/reset",
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.info("Session reset successful")
                return True
            else:
                self.logger.warning(
                    "Session reset failed",
                    status_code=response.status_code
                )
                return False
                
        except Exception as e:
            self.logger.warning("Session reset error", error=str(e))
            return False 