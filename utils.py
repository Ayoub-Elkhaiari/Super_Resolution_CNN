import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import cv2




def create_low_res(images, scale_factor=2):
    low_res_images = []
    for img in images:
        # Downscale the image
        low_res = cv2.resize(img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor))
        # Upscale the image back to original size
        low_res_upscaled = cv2.resize(low_res, (img.shape[1], img.shape[0]))
        low_res_images.append(low_res_upscaled)
    return np.array(low_res_images)


def visualize_samples(lr_images, hr_images, n_samples=5):
    plt.figure(figsize=(10, 4))
    for i in range(n_samples):
        # Low-resolution image
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(lr_images[i])
        plt.axis('off')

        # Corresponding high-resolution image
        plt.subplot(2, n_samples, n_samples + i + 1)
        plt.imshow(hr_images[i])
        plt.axis('off')
    plt.show()
    
    
    
from tensorflow.keras import layers, models

def build_srcnn():
    model = models.Sequential()

    # Feature extraction layer (64 filters, kernel size 9x9)
    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(32, 32, 3)))

    # Non-linear mapping layer (32 filters, kernel size 5x5)
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))

    # Reconstruction layer (3 filters, kernel size 5x5, same as input channels)
    model.add(layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model




# Visualize original low-resolution, predicted high-resolution, and ground truth
def visualize_results(lr_images, predicted_hr, hr_images, n_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        # Low-resolution input
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(lr_images[i])
        plt.title('Low-Res Input')
        plt.axis('off')

        # Predicted high-resolution output
        plt.subplot(3, n_samples, n_samples + i + 1)
        plt.imshow(predicted_hr[i])
        plt.title('Predicted High-Res')
        plt.axis('off')

        # Ground truth high-resolution
        plt.subplot(3, n_samples, 2 * n_samples + i + 1)
        plt.imshow(hr_images[i])
        plt.title('Ground Truth')
        plt.axis('off')

    plt.show()
    
    
    
    
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(ground_truth, predicted):
    return np.mean([psnr(gt, pred) for gt, pred in zip(ground_truth, predicted)])