# Final Project for the course: CUDA Advanced Libraries
- `generate_test_images.py` creates sample test images for processing.
- `gpu_image_processor.py` contains the main logic for GPU-accelerated image processing using PyTorch.
- `README.md` provides an overview and instructions for the project.
- `runLogs.txt` contains logs from running the image processing tasks.
- `data/` directory holds sample input images from when the project was tested.
- `output/` directory is where processed images are saved from the GPU operations from when the project was tested.
- `learningLogs.txt` contains logs from learning and debugging the project.
- `requirements.txt` lists the Python dependencies needed for the project.
## Overview
This project demonstrates GPU-accelerated image processing using PyTorch. It applies various convolutional filters to images, leveraging CUDA for performance improvements. To run this project, ensure you have the required dependencies installed, including PyTorch with CUDA support, Pillow for image handling, and NumPy for numerical operations.
## Instructions
1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Place your input images in the `data/` directory or run the `generate_test_images.py` script to create sample images:
    ```bash
    python generate_test_images.py
    ```
3. Modify the `input` variable in `gpu_image_processor.py` to point to your desired input image file.
4. Run the main script to process an image:
    ```bash
    python gpu_image_processor.py
    ```
5. The processed images will be saved in the `output/` directory.