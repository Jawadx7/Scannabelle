import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

def extract_student_info_regions(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract regions containing student information from the OMR sheet.
    
    Args:
        img: Thresholded image of the OMR sheet
        
    Returns:
        Dictionary containing separate regions for each piece of student information
    """
    # Define relative positions for each information field (to be adjusted based on OMR layout)
    h, w = img.shape[:2]
    regions = {
        'index_number': img[int(0.05*h):int(0.15*h), int(0.1*w):int(0.4*w)],
        'department_code': img[int(0.05*h):int(0.15*h), int(0.45*w):int(0.6*w)],
        'academic_year': img[int(0.05*h):int(0.15*h), int(0.65*w):int(0.8*w)],
        'year_of_study': img[int(0.15*h):int(0.25*h), int(0.1*w):int(0.3*w)],
        'course_code': img[int(0.15*h):int(0.25*h), int(0.35*w):int(0.6*w)],
        'semester': img[int(0.15*h):int(0.25*h), int(0.65*w):int(0.8*w)]
    }
    return regions

def detect_marked_bubbles(region: np.ndarray, num_bubbles: int, threshold: int = 100) -> List[int]:
    """Detect marked bubbles in a region.
    
    Args:
        region: Region containing bubbles
        num_bubbles: Expected number of bubbles
        threshold: Minimum pixel count to consider a bubble marked
        
    Returns:
        List of indices where bubbles are marked (0-based)
    """
    h, w = region.shape[:2]
    bubble_width = w // num_bubbles
    marked_indices = []
    
    for i in range(num_bubbles):
        bubble = region[:, i*bubble_width:(i+1)*bubble_width]
        if cv2.countNonZero(bubble) > threshold:
            marked_indices.append(i)
            
    return marked_indices

def decode_numeric_field(marked_indices: List[int], base: int = 10) -> str:
    """Convert marked bubble indices to numeric string.
    
    Args:
        marked_indices: List of marked bubble indices
        base: Numeric base (10 for decimal)
        
    Returns:
        Decoded numeric string
    """
    return ''.join(str(idx % base) for idx in sorted(marked_indices))

def decode_alpha_field(marked_indices: List[int]) -> str:
    """Convert marked bubble indices to alphabetic string.
    
    Args:
        marked_indices: List of marked bubble indices
        
    Returns:
        Decoded alphabetic string
    """
    return ''.join(chr(65 + idx) for idx in sorted(marked_indices))

def extract_student_details(img: np.ndarray) -> Dict[str, str]:
    """Extract all student details from the OMR sheet.
    
    Args:
        img: Thresholded image of the OMR sheet
        
    Returns:
        Dictionary containing extracted student information
    """
    regions = extract_student_info_regions(img)
    details = {}
    
    # Extract index number (assuming 10 digits)
    index_bubbles = detect_marked_bubbles(regions['index_number'], 10)
    details['index_number'] = decode_numeric_field(index_bubbles)
    
    # Extract department code (assuming 3 letters)
    dept_bubbles = detect_marked_bubbles(regions['department_code'], 3)
    details['department_code'] = decode_alpha_field(dept_bubbles)
    
    # Extract academic year (assuming 4 digits)
    year_bubbles = detect_marked_bubbles(regions['academic_year'], 4)
    details['academic_year'] = decode_numeric_field(year_bubbles)
    
    # Extract year of study (assuming 1 digit)
    study_year_bubbles = detect_marked_bubbles(regions['year_of_study'], 1)
    details['year_of_study'] = decode_numeric_field(study_year_bubbles)
    
    # Extract course code (assuming 6 characters)
    course_bubbles = detect_marked_bubbles(regions['course_code'], 6)
    details['course_code'] = decode_alpha_field(course_bubbles)
    
    # Extract semester (assuming 1 digit)
    semester_bubbles = detect_marked_bubbles(regions['semester'], 1)
    details['semester'] = decode_numeric_field(semester_bubbles)
    
    return details