
import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def gradient_threshold(image, min_threshold=25, max_threshold=120, kernel_size=3):

    # Pretvaranje slike u grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel operator u X smjeru
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=kernel_size
    )

    # Apsolutna vrijednost gradijenta
    abs_sobel_x = np.abs(sobel_x)

    # Skaliranje vrijednosti na opseg 0-255
    max_gradient = np.max(abs_sobel_x)

    if max_gradient == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    scaled_sobel = np.uint8(
        255 * abs_sobel_x / max_gradient
    )

    # Kreiranje binarne slike
    binary = np.zeros_like(scaled_sobel)

    binary[
        (scaled_sobel >= min_threshold) &
        (scaled_sobel <= max_threshold)
    ] = 255

    return binary


def main():

    image_path = PROJECT_DIR / "test_images" / "test1.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image.")
        return

    configurations = [
        (3, 25, 120),
        (3, 20, 180),
        (5, 25, 120),
        (5, 20, 180)
    ]

    output_dir = PROJECT_DIR / "output_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    for kernel_size, min_threshold, max_threshold in configurations:
        binary_image = gradient_threshold(
            image,
            min_threshold,
            max_threshold,
            kernel_size
        )

        output_path = output_dir / (
            f"sobel_k{kernel_size}_t{min_threshold}_{max_threshold}.jpg"
        )

        cv2.imwrite(str(output_path), binary_image)
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
