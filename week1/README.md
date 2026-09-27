# Week 1 — Image Fundamentals with OpenCV

## Overview
Basic image processing pipeline covering grayscale conversion, edge detection,
thresholding, and morphological operations using OpenCV and NumPy.

## What it does
- Reads an image and converts BGR → RGB → Grayscale
- Applies Sobel (X/Y) and Canny edge detection
- Applies binary thresholding and adaptive thresholding
- Applies morphological operations (Erosion, Dilation, Opening, Closing)
  to clean noise and reduce lighting artifacts
- Saves the final cleaned document image using `cv2.imwrite`

## Tools
Python, OpenCV (`opencv-python`), NumPy, Matplotlib

## Input / Output
- Input: any `.jpg` image (tested with a handwritten notebook photo)
- Output: `cleaned_document.jpg`

## Key learnings
- Adaptive thresholding solves uneven lighting better than fixed thresholding
- Morphological kernel size is a trade-off: too small = noise remains,
  too large = fine details (thin text/scratches) get destroyed