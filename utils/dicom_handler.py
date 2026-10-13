import pydicom
import numpy as np
import cv2
from PIL import Image

def load_dicom_or_image(uploaded_file):
    """
    Parses DICOM (.dcm) or standard image files (.png, .jpg)
    and returns a normalized uint8 image array alongside metadata.
    """
    filename = uploaded_file.name.lower()
    
    if filename.endswith(('.dcm', '.dicom')):
        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array.astype(float)
        
        # Rescale intercept/slope if present
        rescale_slope = getattr(dicom, 'RescaleSlope', 1)
        rescale_intercept = getattr(dicom, 'RescaleIntercept', 0)
        pixel_array = pixel_array * rescale_slope + rescale_intercept
        
        # Normalization to 8-bit
        pixel_array -= np.min(pixel_array)
        if np.max(pixel_array) != 0:
            pixel_array /= np.max(pixel_array)
        img_uint8 = (pixel_array * 255).astype(np.uint8)
        
        # Extract metadata
        metadata = {
            "Patient ID": str(getattr(dicom, 'PatientID', 'ANONYMIZED')),
            "Modality": str(getattr(dicom, 'Modality', 'DX/CR')),
            "Pixel Spacing": str(getattr(dicom, 'PixelSpacing', ['0.14', '0.14'])),
            "Window Center": str(getattr(dicom, 'WindowCenter', 'Auto')),
            "Window Width": str(getattr(dicom, 'WindowWidth', 'Auto'))
        }
        return img_uint8, metadata
    else:
        image = Image.open(uploaded_file).convert('L')
        img_uint8 = np.array(image)
        metadata = {
            "Patient ID": "ANONYMIZED_DEMO",
            "Modality": "Standard Image",
            "Pixel Spacing": "N/A",
            "Window Center": "N/A",
            "Window Width": "N/A"
        }
        return img_uint8, metadata

def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)
