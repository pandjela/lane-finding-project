
import cv2
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def color_threshold(image):
    # Pretvaranje slike iz BGR u HLS prostor boja
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # Granice za bijele linije
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 255])

    # Granice za zute linije
    lower_yellow = np.array([15, 30, 100])
    upper_yellow = np.array([35, 204, 255])

    white_mask = cv2.inRange(hls, lower_white, upper_white)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    # Kombinovanje dvije maske
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    return combined_mask


def main():
    image_path = PROJECT_DIR / "test_images" / "test1.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image.")
        return

    binary_image = color_threshold(image)

    output_path = PROJECT_DIR / "output_images" / "color_threshold_test1.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), binary_image)

    print("Color threshold image saved to:", output_path)


if __name__ == "__main__":
    main()
