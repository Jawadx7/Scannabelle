import tkinter as tk
from gui import OMRGraderGUI

def main():
    """Entry point for the OMR Grader application."""
    root = tk.Tk()
    app = OMRGraderGUI(root)
    root.mainloop()
        
    def create_widgets(self):
        # Left Frame (Controls)
        control_frame = tk.Frame(self.root, width=300, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right Frame (Display)
        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Control Frame Components
        tk.Label(control_frame, text="OMR Sheet Grader", font=("Arial", 16), bg="#f0f0f0").pack(pady=20)
        
        # Image Selection
        tk.Label(control_frame, text="Select OMR Sheet:", bg="#f0f0f0").pack(pady=5)
        tk.Entry(control_frame, textvariable=self.image_path, width=30).pack(pady=5)
        tk.Button(control_frame, text="Browse", command=self.browse_image).pack(pady=5)
        
        # Processing Button
        tk.Button(control_frame, text="Grade OMR Sheet", command=self.process_image, 
                 bg="#4CAF50", fg="white").pack(pady=20)
        
        # Results Display
        tk.Label(control_frame, text="Results:", bg="#f0f0f0").pack(pady=5)
        self.results_text = tk.Text(control_frame, height=20, width=35)
        self.results_text.pack(pady=5)
        
        # Display Frame Components (for images)
        self.original_label = tk.Label(display_frame)
        self.original_label.pack(pady=10)
        
        self.processed_label = tk.Label(display_frame)
        self.processed_label.pack(pady=10)
        
        # Add tabs for different views
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(expand=True, fill=tk.BOTH)
        
        self.original_tab = ttk.Frame(self.notebook)
        self.warped_tab = ttk.Frame(self.notebook)
        self.threshold_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.original_tab, text="Original")
        self.notebook.add(self.warped_tab, text="Warped")
        self.notebook.add(self.threshold_tab, text="Threshold")
        
        self.original_display = tk.Label(self.original_tab)
        self.original_display.pack(expand=True, fill=tk.BOTH)
        
        self.warped_display = tk.Label(self.warped_tab)
        self.warped_display.pack(expand=True, fill=tk.BOTH)
        
        self.threshold_display = tk.Label(self.threshold_tab)
        self.threshold_display.pack(expand=True, fill=tk.BOTH)
    
    def browse_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filepath:
            self.image_path.set(filepath)
            self.display_original_image(filepath)
    
    def display_original_image(self, path):
        img = Image.open(path)
        img.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(img)
        
        self.original_display.config(image=photo)
        self.original_display.image = photo
    
    def display_processed_images(self, original, warped, threshold):
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
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            # Your OMR processing code (slightly modified for the GUI)
            path = self.image_path.get()
            img = cv2.imread(path)
            if img is None:
                raise Exception("Could not load image")
            
            widthImg, heightImg = 600, 700
            img = cv2.resize(img, (widthImg, heightImg))
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
            imgCanny = cv2.Canny(imgBlur, 10, 50)
            
            contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectCon = self.rectContours(contours)
            
            if len(rectCon) < 1:
                raise Exception("No rectangular contours found")
            
            biggestContour = self.getCornerPoints(rectCon[0])
            if biggestContour is None or len(biggestContour) != 4:
                raise Exception("Biggest contour not valid")
            
            biggestContour = self.reorder(biggestContour)
            
            # Perspective Transform
            pts1 = np.float32(biggestContour)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            
            # Thresholding
            imgWarpGray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
            _, imgThresh = cv2.threshold(imgWarpGray, 218, 250, cv2.THRESH_BINARY_INV)
            
            # Process answer boxes
            
            boxes = self.splitBoxes(imgThresh)
            answers = self.findShadedAnswers(boxes)
            
            # Display results
            self.display_results(answers)
            self.display_processed_images(img, imgWarp, imgThresh)
            
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
    
    def display_results(self, answers):
        self.results_text.delete(1.0, tk.END)
        for i, ans in enumerate(answers):
            result = f"Question {i+1}: {'ABCDE'[ans] if ans != -1 else 'No answer'}\n"
            self.results_text.insert(tk.END, result)
    
    # Your OMR processing functions (copied from your code)
    def rectContours(self, contours):
        rectCon = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > 1000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if len(approx) == 4:
                    rectCon.append(i)
        rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
        return rectCon
    
    def getCornerPoints(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        return approx
    
    def reorder(self, points):
        points = points.reshape((4, 2))
        newPoints = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        diff = np.diff(points, axis=1)
        
        newPoints[0] = points[np.argmin(add)]      # Top-left
        newPoints[3] = points[np.argmax(add)]      # Bottom-right
        newPoints[1] = points[np.argmin(diff)]     # Top-right
        newPoints[2] = points[np.argmax(diff)]     # Bottom-left
        return newPoints
    
    def splitBoxes(self, img, rows=20, cols=5):
        boxes = []
        h, w = img.shape[:2]
        
        # Calculate exact split sizes
        row_height = h // rows
        col_width = w // cols
        
        # Handle remainder by adding 1 pixel to some splits
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
    
    def findShadedAnswers(self, boxes):
        answers = []
        for q in range(0, 20):
            currentBoxes = boxes[q*5:(q+1)*5]
            pixelCounts = [cv2.countNonZero(box) for box in currentBoxes]
            marked = np.argmax(pixelCounts)
            if pixelCounts[marked] > 500:  # adjustable threshold
                answers.append(marked)
            else:
                answers.append(-1)  # Not marked properly
        return answers

if __name__ == "__main__":
    main()