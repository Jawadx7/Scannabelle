from typing import Dict, List
import csv
import os

class AnswerManager:
    def __init__(self):
        """Initialize the answer manager."""
        self.answers: Dict[str, str] = {}
        self.csv_file = 'correct_answers.csv'

    def set_answer(self, question_num: int, answer: str) -> None:
        """Set the correct answer for a question.
        
        Args:
            question_num: Question number (1-based)
            answer: Answer choice (A-E)
        """
        if not (1 <= question_num <= 100):
            raise ValueError("Question number must be between 1 and 100")
        if answer not in 'ABCDE':
            raise ValueError("Answer must be A, B, C, D, or E")
        
        self.answers[f"Q{question_num}"] = answer

    def get_answer(self, question_num: int) -> str:
        """Get the correct answer for a question.
        
        Args:
            question_num: Question number (1-based)
            
        Returns:
            The correct answer (A-E) or empty string if not set
        """
        return self.answers.get(f"Q{question_num}", "")

    def get_all_answers(self) -> Dict[str, str]:
        """Get all stored correct answers.
        
        Returns:
            Dictionary of question numbers to answers
        """
        return self.answers.copy()

    def clear_answers(self) -> None:
        """Clear all stored answers."""
        self.answers.clear()

    def save_to_csv(self) -> None:
        """Save current answers to CSV file."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Answer'])
            for q_num in range(1, 101):
                answer = self.get_answer(q_num)
                if answer:  # Only save questions that have answers
                    writer.writerow([f"Q{q_num}", answer])

    def load_from_csv(self) -> bool:
        """Load answers from CSV file.
        
        Returns:
            True if file was loaded successfully, False if file doesn't exist
        """
        if not os.path.exists(self.csv_file):
            return False

        self.clear_answers()
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_num = int(row['Question'].lstrip('Q'))
                self.set_answer(q_num, row['Answer'])
        return True

    def get_grading_list(self) -> List[int]:
        """Convert stored answers to format needed by grader.
        
        Returns:
            List of integers (0-4 for A-E) for grading
        """
        answer_list = []
        for q_num in range(1, 21):  # Only first 20 questions for grading
            answer = self.get_answer(q_num)
            if answer:
                # Convert A-E to 0-4
                answer_list.append(ord(answer) - ord('A'))
            else:
                answer_list.append(0)  # Default to A if not set
        return answer_list