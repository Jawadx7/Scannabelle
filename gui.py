import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from omr_processing import image_utils, bubble_detector, grader

class OMRGraderGUI:
    def __init__(self, root: tk.Tk):
        """Initialize the OMR Grader GUI.
        
        Args:
            root: The root Tkinter window
        """
        self.root = root
        self.root.title("OMR Sheet Grading System")
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path = tk.StringVar()
        self.results = []
        
        # Create GUI components
        self.create_widgets()
    
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Left Frame (Controls)
        control_frame = tk.Frame(self.root, width=300, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right Frame (Display)
        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self._create_control_panel(control_frame)
        self._create_display_panel(display_frame)
    
    def _create_control_panel(self, parent: tk.Frame):
        """Create the control panel widgets.
        
        Args:
            parent: Parent frame for control panel
        """
        # Title
        tk.Label(parent, text="OMR Sheet Grader", 
                font=("Arial", 16), bg="#f0f0f0").pack(pady=20)
        
        # Image Selection
        tk.Label(parent, text="Select OMR Sheet:", 
                bg="#f0f0f0").pack(pady=5)
        tk.Entry(parent, textvariable=self.image_path, 
                width=30).pack(pady=5)
        tk.Button(parent, text="Browse", 
                command=self.browse_image).pack(pady=5)
        
        # Processing Button
        tk.Button(parent, text="Grade OMR Sheet", 
                command=self.process_image,
                bg="#4CAF50", fg="white").pack(pady=20)
        
        # Results Display
        tk.Label(parent, text="Results:", bg="#f0f0f0").pack(pady=5)
        self.results_text = tk.Text(parent, height=20, width=35)
        self.results_text.pack(pady=5)
    
    def _create_display_panel(self, parent: tk.Frame):
        """Create the image display panel with tabs.
        
        Args:
            parent: Parent frame for display panel
        """
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(expand=True, fill=tk.BOTH)
        
        # Create tabs
        self.original_tab = ttk.Frame(self.notebook)
        self.warped_tab = ttk.Frame(self.notebook)
        self.threshold_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.original_tab, text="Original")
        self.notebook.add(self.warped_tab, text="Warped")
        self.notebook.add(self.threshold_tab, text="Threshold")
        
        # Create display labels
        self.original_display = tk.Label(self.original_tab)
        self.original_display.pack(expand=True, fill=tk.BOTH)
        
        self.warped_display = tk.Label(self.warped_tab)
        self.warped_display.pack(expand=True, fill=tk.BOTH)
        
        self.threshold_display = tk.Label(self.threshold_tab)
        self.threshold_display.pack(expand=True, fill=tk.BOTH)
    
    def browse_image(self):
        """Open file dialog to select an image file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filepath:
            self.image_path.set(filepath)
            self.display_original_image(filepath)
    
    def display_original_image(self, path: str):
        """Display the original image in the first tab.
        
        Args:
            path: Path to the image file
        """
        img = Image.open(path)
        img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(img)
        
        self.original_display.config(image=photo)
        self.original_display.image = photo
    
    def display_processed_images(self, original: np.ndarray, 
                               warped: np.ndarray, 
                               threshold: np.ndarray):
        """Display processed images in their respective tabs.
        
        Args:
            original: Original processed image
            warped: Perspective transformed image
            threshold: Thresholded image
        """
        # Original Image
        img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(img)
        self.original_display.config(image=photo)
        self.original_display.image = photo
        
        # Warped Image
        img_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        img_warped = Image.fromarray(img_warped)
        img_warped.thumbnail((600, 600))
        photo_warped = ImageTk.PhotoImage(img_warped)
        self.warped_display.config(image=photo_warped)
        self.warped_display.image = photo_warped
        
        # Threshold Image
        img_thresh = Image.fromarray(threshold)
        img_thresh.thumbnail((600, 600))
        photo_thresh = ImageTk.PhotoImage(img_thresh)
        self.threshold_display.config(image=photo_thresh)
        self.threshold_display.image = photo_thresh
    
    def process_image(self):
        """Process the selected image and display results."""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            # Process image using utility functions
            img_canny = image_utils.load_and_preprocess_image(self.image_path.get())
            if img_canny is None:
                raise Exception("Could not load image")
            
            # Find and process contours
            rect_contours = image_utils.find_rectangle_contours(img_canny)
            if not rect_contours:
                raise Exception("No rectangular contours found")
            
            # Get and validate corner points
            biggest_contour = image_utils.get_corner_points(rect_contours[0])
            if biggest_contour is None:
                raise Exception("Biggest contour not valid")
            
            # Reorder points and apply perspective transform
            ordered_points = image_utils.reorder_points(biggest_contour)
            img = cv2.imread(self.image_path.get())
            img = cv2.resize(img, (600, 700))
            warped = image_utils.apply_perspective_transform(img, ordered_points, 600, 700)
            
            # Threshold the image
            thresh = image_utils.threshold_image(warped)
            
            # Detect answers
            answers, boxes = bubble_detector.analyze_answer_sheet(thresh)
            
            # For demo purposes, using a dummy correct answer key
            # In practice, this would be loaded from a configuration or database
            correct_answers = [0] * 20  # All A's for demonstration
            
            # Grade the answers
            grade_results = grader.grade_answers(answers, correct_answers)
            result_strings = grader.format_results(grade_results, answers, correct_answers)
            
            # Display results
            self.display_results(result_strings)
            self.display_processed_images(img, warped, thresh)
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
    
    def display_results(self, results: List[str]):
        """Display grading results in the text widget.
        
        Args:
            results: List of result strings to display
        """
        self.results_text.delete(1.0, tk.END)
        for result in results:
            self.results_text.insert(tk.END, result + '\n')