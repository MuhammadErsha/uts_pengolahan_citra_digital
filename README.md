# uts_pengolahan_citra_digital
nama:muhammad ersha rahmadani
nim:23422047
kelas tif pagi 22 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk histogram equalization
def histogram_equalization(image):
    # Mengambil dimensi gambar
    height, width = image.shape

    # Hitung histogram
    histogram = [0] * 256
    for i in range(height):
        for j in range(width):
            intensity = image[i, j]
            histogram[intensity] += 1

    # Hitung histogram kumulatif
    cumulative_histogram = [0] * 256
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    # Normalisasi histogram kumulatif
    total_pixels = height * width
    normalized_histogram = [(cumulative_histogram[i] * 255) // total_pixels for i in range(256)]

    # Buat gambar hasil equalization
    equalized_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = normalized_histogram[image[i, j]]

    return equalized_image, histogram, normalized_histogram

# Baca gambar dalam mode grayscale
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Lakukan histogram equalization
equalized_image, original_histogram, equalized_histogram = histogram_equalization(image)

# Visualisasi hasil
plt.figure(figsize=(12, 8))

# Gambar asli
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Histogram asli
plt.subplot(2, 2, 2)
plt.bar(range(256), original_histogram, color='black')
plt.title("Original Histogram")

# Gambar hasil equalization
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

# Histogram hasil equalization
plt.subplot(2, 2, 4)
plt.bar(range(256), equalized_histogram, color='black')
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()
