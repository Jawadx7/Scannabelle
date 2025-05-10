# OMR Sheet Grading System

A Python-based Optical Mark Recognition (OMR) system for grading multiple-choice answer sheets. This application provides a user-friendly GUI interface for processing and grading OMR sheets with support for 20 questions, each having 5 options (A-E).

## Features

- User-friendly GUI interface built with Tkinter
- Real-time image processing and answer detection
- Support for 20 multiple-choice questions (A-E options)
- Visual feedback with original, warped, and thresholded image views
- Detailed grading results with score calculation
- Error handling and input validation

## Project Structure

```
├── main.py                 # Application entry point
├── gui.py                  # GUI implementation
├── omr_processing/         # Core OMR processing modules
│   ├── image_utils.py      # Image processing utilities
│   ├── bubble_detector.py  # Answer bubble detection
│   └── grader.py          # Answer grading logic
```

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Jawadx7/Scannabelle
cd omr-grader
```

2. Install required packages:

```bash
pip install opencv-python numpy pillow
```

## Usage

1. Run the application:

```bash
python main.py
```

2. Use the GUI to:
   - Click "Browse" to select an OMR sheet image
   - Click "Grade OMR Sheet" to process the image
   - View results in the results panel
   - Check different image views in the tabs

## Image Requirements

- Clear, well-lit images of OMR sheets
- Visible answer bubbles with good contrast
- Sheet should be the main focus of the image
- Supported formats: JPG, JPEG, PNG

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
