from utils import build_srcnn, visualize_results, visualize_samples, create_low_res, calculate_psnr
import tensorflow as tf 
import numpy as np


# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to [0, 1] range
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0



# Create low-resolution versions of training and testing data
x_train_lr = create_low_res(x_train)
x_test_lr = create_low_res(x_test)



# Show some samples
visualize_samples(x_train_lr, x_train)


# Build the SRCNN model
srcnn = build_srcnn()
srcnn.summary()


# Train the SRCNN model
history = srcnn.fit(x_train_lr, x_train,
                    validation_data=(x_test_lr, x_test),
                    epochs=10,
                    batch_size=64)



# Predict on test images
predicted = srcnn.predict(x_test_lr)


# Show results for a few test samples
visualize_results(x_test_lr, predicted, x_test)

psnr_value = calculate_psnr(x_test, predicted)
print(f"Average PSNR value for Predicted: {psnr_value:.2f} dB")
psnr_value = calculate_psnr(x_test, predicted)
print(f"Average PSNR value for the input: {psnr_value:.2f} dB")
