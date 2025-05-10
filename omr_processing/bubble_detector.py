import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from . import student_info_detector

def split_answer_boxes(img: np.ndarray, rows: int = 20, cols: int = 5) -> List[np.ndarray]:
    """Split the thresholded image into individual answer boxes.
    
    Args:
        img: Input thresholded image
        rows: Number of questions
        cols: Number of options per question
        
    Returns:
        List of individual answer box images
    """
    boxes = []
    h, w = img.shape[:2]
    
    # Calculate exact split sizes
    row_height = h // rows
    col_width = w // cols
    
    # Handle remainder pixels
    row_remainder = h % rows
    col_remainder = w % cols
    
    # Split rows
    row_splits = []
    current_y = 0
    for i in range(rows):
        split_height = row_height + (1 if i < row_remainder else 0)
        row_splits.append(img[current_y:current_y+split_height, 0:w])
        current_y += split_height
    
    # Split columns for each row
    for row in row_splits:
        current_x = 0
        for j in range(cols):
            split_width = col_width + (1 if j < col_remainder else 0)
            box = row[0:row_height, current_x:current_x+split_width]
            boxes.append(box)
            current_x += split_width
    
    return boxes

def detect_marked_answers(boxes: List[np.ndarray], threshold: int = 500) -> List[int]:
    """Detect which answer bubbles are marked for each question.
    
    Args:
        boxes: List of answer box images
        threshold: Minimum pixel count to consider an answer marked
        
    Returns:
        List of detected answers (-1 for unmarked questions)
    """
    answers = []
    for q in range(0, len(boxes) // 5):
        current_boxes = boxes[q*5:(q+1)*5]
        pixel_counts = [cv2.countNonZero(box) for box in current_boxes]
        marked = np.argmax(pixel_counts)
        
        # Check if the marked answer meets the threshold
        if pixel_counts[marked] > threshold:
            answers.append(marked)
        else:
            answers.append(-1)  # Not marked properly
    
    return answers

def validate_answer_boxes(boxes: List[np.ndarray], expected_questions: int = 20) -> bool:
    """Validate that we have the correct number of answer boxes.
    
    Args:
        boxes: List of answer box images
        expected_questions: Expected number of questions
        
    Returns:
        True if the number of boxes is correct, False otherwise
    """
    return len(boxes) == expected_questions * 5  # 5 options per question

def analyze_answer_sheet(img: np.ndarray) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Analyze an answer sheet image and return detected student info and answers.
    
    Args:
        img: Preprocessed and thresholded image
        
    Returns:
        Tuple of (student details, answer dictionary)
    """
    # Extract student information
    student_details = student_info_detector.extract_student_details(img)
    
    # Extract answer section (assuming it starts after student info section)
    h, w = img.shape[:2]
    answer_section = img[int(0.3*h):, :]
    
    # Process answers
    boxes = split_answer_boxes(answer_section)
    if not validate_answer_boxes(boxes):
        raise ValueError("Invalid number of answer boxes detected")
    
    marked_answers = detect_marked_answers(boxes)
    
    # Convert numeric answers to letter format
    answer_dict = {}
    for i, answer in enumerate(marked_answers):
        if answer != -1:
            answer_dict[f"Q{i+1}"] = chr(65 + answer)  # Convert 0-4 to A-E
    
    return student_details, answer_dict