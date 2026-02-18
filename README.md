# Offroad-Terrain-Segmentation-DINOv
1. Overview
This project implements a binary semantic segmentation model designed for autonomous offroad navigation. It specifically identifies "Dry Grass" paths (Class 1) against all other terrain clutter (Class 0). The model leverages the DINOv2-Small backbone for high-density feature extraction and a custom ConvNeXt-style segmentation head for efficient inference.

2. Environment & Dependency Requirements
The model was developed and tested on an NVIDIA RTX 4050 (Laptop GPU) using CUDA 12.x.

Required Libraries:

torch & torchvision (Core Deep Learning)

numpy (Numerical operations)

opencv-python (Visualizations)

Pillow (Image processing)

tqdm (Progress tracking)

matplotlib (Metrics and plots)

3. Step-by-Step Instructions to Run
To reproduce the evaluation and test the model, follow these steps:

Isolate Environment: Ensure your terminal is running in your dedicated Conda or virtual environment.

Verify File Structure: Ensure the segmentation_head.pth weights and the Offroad_Segmentation_testImages folder are in the same directory as the script.

Execute Evaluation: Run the following command in your terminal:

DOS
python test_segmentation.py
Note: The script is optimized with absolute Windows paths to prevent FileNotFoundError issues during batch processing.

4. Reproducing Final Results
The following results were achieved during the final validation and testing phase:

Validation Mean IoU: 0.6185

Final Test IoU: 0.2801

Inference Speed: 10.79 iterations/second (on RTX 4050)

5. Expected Outputs & Interpretation
Upon completion, the script generates a summary in the terminal and populates the ./test_results folder:

Final Test IoU: This is the primary accuracy metric. A score of 0.2801 indicates that the model generalizes to unseen offroad data without having been exposed to these specific frames during training.

Visual Comparisons: Check the comparisons/ subfolder for side-by-side images (Input vs. Ground Truth vs. Prediction).

Inference Speed: The 10.79 it/s result confirms the model is viable for real-time deployment on offroad vehicle hardware.
