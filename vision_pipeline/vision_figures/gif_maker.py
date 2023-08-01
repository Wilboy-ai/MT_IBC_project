import numpy as np
import cv2
import os
import natsort

# Set the paths to the image folders
folder1_path = 'Segmented'
folder2_path = 'OnlyNeedle'


# Get the list of files in both folders
folder1_images = sorted(os.listdir(folder1_path), key=lambda x: int(x.split('-')[1].split('.')[0]))
folder2_images = sorted(os.listdir(folder2_path), key=lambda x: int(x.split('-')[1].split('.')[0]))


# Create an output folder to save the concatenated images
output_folder = 'final'
os.makedirs(output_folder, exist_ok=True)

# Iterate over the image pairs
for img1, img2 in zip(folder1_images, folder2_images):
    # Read the images
    img1_path = os.path.join(folder1_path, img1)
    img2_path = os.path.join(folder2_path, img2)

    print(f'Concat images {img1} and {img2}')

    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)

    # Concatenate the images vertically
    concatenated_image = cv2.vconcat([image1, image2])

    # Save the concatenated image
    output_path = os.path.join(output_folder, f'concatenated_{img1}')
    cv2.imwrite(output_path, concatenated_image)

    print(f"Concatenated image saved: {output_path}")

