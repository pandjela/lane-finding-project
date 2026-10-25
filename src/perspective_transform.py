import cv2
import numpy as np


def perspective_transform(image, src, dst):

    # Convert source and destination points to float32
    src = np.float32(src)
    dst = np.float32(dst)

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Calculate the inverse matrix to restore the original perspective
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Get image dimensions
    height, width = image.shape[:2]

    # Apply the perspective transformation
    warped = cv2.warpPerspective(
        image,
        M,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

    return warped, M, Minv