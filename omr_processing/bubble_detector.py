import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
# from . import student_info_detector

def split_answer_boxes(img: np.ndarray, rows: int = 30, cols: int = 5) -> List[np.ndarray]:
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

def detect_marked_answers(boxes: List[np.ndarray], threshold_ratio: float = 0.3) -> List[int]:
    """Detect which answer bubbles are marked for each question using adaptive thresholding.
    
    Args:
        boxes: List of answer box images
        threshold_ratio: Ratio of max pixel count to consider an answer marked
        
    Returns:
        List of detected answers (-1 for unmarked questions)
    """
    answers = []
    for q in range(0, len(boxes) // 5):
        current_boxes = boxes[q*5:(q+1)*5]
        
        # Apply adaptive thresholding to each box
        processed_boxes = []
        for box in current_boxes:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(box, (3, 3), 0)
            # Apply adaptive thresholding
            adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            processed_boxes.append(adaptive)
        
        # Count non-zero pixels
        pixel_counts = [cv2.countNonZero(box) for box in processed_boxes]
        max_count = max(pixel_counts)
        marked = np.argmax(pixel_counts)
        
        # Use relative thresholding
        if max_count > 0 and pixel_counts[marked] > max_count * threshold_ratio:
            # Verify this is significantly higher than other options
            sorted_counts = sorted(pixel_counts, reverse=True)
            if len(sorted_counts) > 1 and sorted_counts[0] > sorted_counts[1] * 1.2:  # 20% higher than next highest
                answers.append(marked)
            else:
                answers.append(-1)  # Multiple answers or unclear marking
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

def analyze_answer_sheet(img: np.ndarray) -> Dict[str, str]:
    """Analyze an answer sheet image and return detected answers.
    
    Args:
        img: Preprocessed and thresholded image
        
    Returns:
        Dictionary mapping question numbers to letter answers (A-E)
    """
    # Process answers
    boxes = split_answer_boxes(img)
    print(f"Detected {len(boxes)} answer boxes")
    # print("Boxes", boxes)
    if not validate_answer_boxes(boxes):
        raise ValueError("Invalid number of answer boxes detected")
    
    marked_answers = detect_marked_answers(boxes)
    
    # Convert numeric answers to letter format
    answer_dict = {}
    for i, answer in enumerate(marked_answers):
        if answer != -1:
            answer_dict[f"Q{i+1}"] = chr(65 + answer)  # Convert 0-4 to A-E
    
    return answer_dict


def extract_answers(warped_thresh: np.ndarray, num_questions: int = 20, num_choices: int = 5) -> dict:
    """Extract answers from the thresholded image.
    
    Args:
        warped_thresh: Thresholded image
        num_questions: Number of questions
        num_choices: Number of choices per question
        
    Returns:
        Dictionary mapping question numbers to selected answers
    """
    # After processing all questions and determining answers
    answers = {}
    for q in range(num_questions):
        # Add the answer to the dictionary
        question_key = f"Q{q+1}"
        answers[question_key] = selected_answer  # This might be a letter A-E or None
    
    # Print the extracted answers alongside question numbers
    print("\n===== EXTRACTED STUDENT ANSWERS =====")
    for q_num, answer in answers.items():
        print(f"{q_num}: {answer if answer is not None else 'No answer detected'}")
    print("====================================\n")
    
    return answers