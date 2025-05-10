from typing import List, Dict, Tuple

def grade_answers(student_answers: List[int], correct_answers: List[int]) -> Dict[str, float]:
    """Grade student answers against correct answers.
    
    Args:
        student_answers: List of student's answers (0-4 for A-E, -1 for unmarked)
        correct_answers: List of correct answers (0-4 for A-E)
        
    Returns:
        Dictionary containing score information
    """
    # Use the number of correct answers as the total questions to grade
    total_questions = len(correct_answers)
    
    # Ensure student_answers is at least as long as correct_answers by padding with -1
    if len(student_answers) < total_questions:
        student_answers = student_answers + [-1] * (total_questions - len(student_answers))
    
    # Only grade up to the number of correct answers provided
    student_answers = student_answers[:total_questions]
    
    correct_count = sum(1 for s, c in zip(student_answers, correct_answers) if s == c)
    unanswered = sum(1 for ans in student_answers if ans == -1)
    incorrect = total_questions - correct_count - unanswered
    
    score_percentage = (correct_count / total_questions) * 100
    
    return {
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "incorrect_answers": incorrect,
        "unanswered": unanswered,
        "score_percentage": score_percentage
    }

def format_results(grading_results: Dict[str, float], student_answers: List[int], 
                  correct_answers: List[int]) -> List[str]:
    """Format grading results into human-readable strings.
    
    Args:
        grading_results: Dictionary containing grading information
        student_answers: List of student's answers
        correct_answers: List of correct answers
        
    Returns:
        List of formatted result strings
    """
    results = [
        f"Total Questions: {grading_results['total_questions']}",
        f"Correct Answers: {grading_results['correct_answers']}",
        f"Incorrect Answers: {grading_results['incorrect_answers']}",
        f"Unanswered: {grading_results['unanswered']}",
        f"Score: {grading_results['score_percentage']:.2f}%",
        "\nDetailed Results:"
    ]
    
    for i, (student, correct) in enumerate(zip(student_answers, correct_answers)):
        status = "✓" if student == correct else "✗"
        student_ans = 'ABCDE'[student] if student != -1 else 'None'
        correct_ans = 'ABCDE'[correct]
        results.append(
            f"Q{i+1}: {status} Your Answer: {student_ans} | Correct: {correct_ans}"
        )
    
    return results

def analyze_common_mistakes(student_answers: List[int], 
                          correct_answers: List[int]) -> Dict[str, int]:
    """Analyze patterns in incorrect answers.
    
    Args:
        student_answers: List of student's answers
        correct_answers: List of correct answers
        
    Returns:
        Dictionary containing mistake patterns
    """
    patterns = {
        "skipped_answers": 0,
        "wrong_by_one": 0,  # Selected option next to correct one
        "opposite_end": 0,  # Selected option at opposite end
        "middle_bias": 0    # Incorrectly selected middle option (C)
    }
    
    for student, correct in zip(student_answers, correct_answers):
        if student == -1:
            patterns["skipped_answers"] += 1
        elif student != correct:
            if abs(student - correct) == 1:
                patterns["wrong_by_one"] += 1
            elif abs(student - correct) >= 3:
                patterns["opposite_end"] += 1
            elif student == 2:  # Option C (middle)
                patterns["middle_bias"] += 1
    
    return patterns