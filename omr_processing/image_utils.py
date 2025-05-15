import cv2
import numpy as np
from typing import List, Tuple, Optional

def load_and_preprocess_image(image_path: str, width: int = 600, height: int = 700) -> Optional[np.ndarray]:
    """Load and preprocess the image for OMR processing.
    
    Args:
        image_path: Path to the image file
        width: Desired width of the processed image
        height: Desired height of the processed image
        
    Returns:
        Preprocessed image or None if loading fails
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (width, height))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    return img_canny

def find_rectangle_contours(img: np.ndarray, min_area: float = 1000) -> List[np.ndarray]:
    """Find and sort rectangular contours in the image.
    
    Args:
        img: Input image
        min_area: Minimum contour area to consider
        
    Returns:
        List of rectangular contours sorted by area
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rect_contours.append(contour)
                
    return sorted(rect_contours, key=cv2.contourArea, reverse=True)

def get_corner_points(contour: np.ndarray) -> Optional[np.ndarray]:
    """Get corner points from a contour.
    
    Args:
        contour: Input contour
        
    Returns:
        Corner points or None if invalid
    """
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return approx if len(approx) == 4 else None

def reorder_points(points: np.ndarray) -> np.ndarray:
    """Reorder points in clockwise order (top-left, top-right, bottom-left, bottom-right).
    
    Args:
        points: Input points array
        
    Returns:
        Reordered points
    """

    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.float32)
    
    add = points.sum(1)
    diff = np.diff(points, axis=1)
    
    new_points[0] = points[np.argmin(add)]  # Top-left
    new_points[1] = points[np.argmin(diff)] # Top-right
    new_points[2] = points[np.argmax(diff)] # Bottom-left
    new_points[3] = points[np.argmax(add)]  # Bottom-right

    return new_points
    

def apply_perspective_transform(img: np.ndarray, points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply perspective transform to get a top-down view of the OMR sheet.
    
    Args:
        img: Input image
        points: Corner points for perspective transform
        width: Output image width
        height: Output image height
        
    Returns:
        Transformed image
    """
    pts1 = np.float32(points)
    pts2 = np.float32([
    [0, 0],           # Top-left
    [width, 0],       # Top-right
    [0, height],      # Bottom-left
    [width, height]   # Bottom-right
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (width, height))

def threshold_image(img: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to the image for better bubble detection.
    
    Args:
        img: Input image
        
    Returns:
        Thresholded image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    # Apply Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    # Apply adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return img_thresh


def load_and_preprocess_image_from_array(img: np.ndarray, width: int = 600, height: int = 700) -> Optional[np.ndarray]:
    img = cv2.resize(img, (width, height))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    return img_canny
