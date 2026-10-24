import cv2
import numpy as np


def region_of_interest(binary_image, vertices):
    # Create a black mask with the same dimensions as the input image
    mask = np.zeros_like(binary_image)

    # Fill the region of interest with white
    cv2.fillPoly(
        mask,
        [np.array(vertices, dtype=np.int32)],
        255
    )

    # Keep only the pixels inside the region of interest
    masked_image = cv2.bitwise_and(binary_image, mask)

    return masked_image