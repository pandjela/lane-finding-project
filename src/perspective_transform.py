
import cv2
import numpy as np


def perspective_transform(image, src, dst):
    
    src = np.float32(src)
    dst = np.float32(dst)

    # Matrica perspektivne transformacije
    M = cv2.getPerspectiveTransform(src, dst)

    # Inverzna matrica za kasnije vracanje na originalnu sliku
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Dimenzije slike
    height, width = image.shape[:2]

    # Transformacija slike
    warped = cv2.warpPerspective(
        image,
        M,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

    return warped, M, Minv
