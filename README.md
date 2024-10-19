# SRCNN Image Super-Resolution

A deep learning project implementing Super-Resolution Convolutional Neural Network (SRCNN) to enhance the resolution of images using the CIFAR-10 dataset. This model can upscale low-resolution images while preserving and enhancing important details.

## Project Overview

This project implements a Super-Resolution CNN that:
1. Downsamples high-resolution images from CIFAR-10
2. Trains a CNN model to reconstruct high-resolution images
3. Evaluates the results using PSNR (Peak Signal-to-Noise Ratio)
4. Provides visualization tools for comparing results

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- OpenCV (cv2)
- Matplotlib
- scikit-image

Install dependencies using:
```bash
pip install tensorflow numpy opencv-python matplotlib scikit-image
```

## Project Structure

```
Super_Resolution_CNN/
├── utils.py          # Utility functions and model architecture
├── main.py          # Main training and evaluation script
└── README.md        # Project documentation
```

## Features

- Image downsampling with configurable scale factor
- SRCNN architecture implementation
- Real-time visualization of:
  - Original high-resolution images
  - Generated low-resolution images
  - Super-resolved predictions
- PSNR calculation for quality assessment
- Training progress monitoring

## Implementation Details

### Model Architecture (SRCNN)

The model consists of three convolutional layers:
1. Feature Extraction
   - 64 filters of size 9×9
   - ReLU activation
2. Non-linear Mapping
   - 32 filters of size 5×5
   - ReLU activation
3. Reconstruction
   - 3 filters of size 1×1 (RGB channels)
   - Sigmoid activation

### Training Configuration
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Batch Size: 64
- Epochs: 10
- Input Shape: 32x32x3 (CIFAR-10 images)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Ayoub-Elkhaiari/Super_Resolution_CNN.git
cd Super_Resolution_CNN
```

2. Run the main script:
```bash
python main.py
```

The script will:
- Load and preprocess CIFAR-10 dataset
- Create low-resolution versions of images
- Train the SRCNN model
- Display results and PSNR metrics

## Key Functions

### In utils.py:

1. `create_low_res(images, scale_factor=2)`
   - Creates low-resolution versions of input images
   - Parameters:
     - images: Input high-resolution images
     - scale_factor: Downsampling factor (default: 2)

2. `build_srcnn()`
   - Constructs the SRCNN model architecture
   - Returns compiled model ready for training

3. `visualize_results(lr_images, predicted_hr, hr_images, n_samples=5)`
   - Displays comparison between low-res, predicted, and ground truth images

4. `calculate_psnr(ground_truth, predicted)`
   - Calculates average PSNR between ground truth and predicted images

## Customization

You can modify the following parameters:

In `utils.py`:
- Scale factor for resolution reduction
- Model architecture parameters
- Visualization settings

In `main.py`:
- Training parameters (epochs, batch_size)
- Number of samples to visualize
- PSNR calculation parameters

## Results

The project provides:
1. Visual comparisons between:
   - Original low-resolution images
   - Super-resolved images
   - Ground truth high-resolution images
2. PSNR metrics for quality assessment
3. Training history with loss and accuracy metrics

4. Example of the results:
   ![Screenshot 2024-10-20 010455](https://github.com/user-attachments/assets/dd93d973-acf5-43ff-87a8-1b065950597e)

   PSNR for predicted and the true images of CIFAR10:
   ![Screenshot 2024-10-20 011038](https://github.com/user-attachments/assets/d8bc40bf-1b44-4203-8886-3221f46ce020)

   PSNR for the input images (low res) and the true images of CIFAR10:
   ![Screenshot 2024-10-20 011045](https://github.com/user-attachments/assets/e030516a-977f-4795-b1a2-3c23296a2e6e)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


## Acknowledgments

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- Original SRCNN paper: "Image Super-Resolution Using Deep Convolutional Networks"
- Link to Paper: https://arxiv.org/pdf/1501.00092

## Performance Metrics

The model's performance is evaluated using:
- PSNR (Peak Signal-to-Noise Ratio)
- Visual comparison of image quality
- Training/validation loss curves
