
import cv2
from pathlib import Path
import numpy as np

from camera_calibration import collect_calibration_points
from color_threshold import color_threshold
from gradient_threshold import gradient_threshold
from perspective_transform import perspective_transform

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    #  Prikupljanje kalibracionih tacaka
    object_points, image_points, image_size = collect_calibration_points()

    if len(image_points) < 3 or image_size is None:
        print("Not enough images for calibration.")
        return

    # Kalibracija kamere
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    print("Calibration RMS error:", rms)

    test_dir = PROJECT_DIR / "test_images"
    output_dir = PROJECT_DIR / "output_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prolazak kroz sve testne slike
    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not load:", image_path.name)
            continue

        # Dimenzije trenutne fotografije
        height, width = image.shape[:2]

        # Dimenzije fotografija koriscenih za kalibraciju
        calib_width, calib_height = image_size

        # Faktori skaliranja
        scale_x = width / calib_width
        scale_y = height / calib_height

        # Prilagodjavanje matrice kamere
        scaled_camera_matrix = camera_matrix.copy()

        scaled_camera_matrix[0, 0] *= scale_x
        scaled_camera_matrix[0, 2] *= scale_x
        scaled_camera_matrix[1, 1] *= scale_y
        scaled_camera_matrix[1, 2] *= scale_y

        # Ispravljanje distorzije
        undistorted = cv2.undistort(
            image,
            scaled_camera_matrix,
            dist_coeffs,
            None,
            scaled_camera_matrix
        )

        # Color i Gradient Threshold
        color_binary = color_threshold(undistorted)
        gradient_binary = gradient_threshold(undistorted)

        # Kombinovanje rezultata
        combined_binary = cv2.bitwise_or(
            color_binary,
            gradient_binary
        )

        # Perspective Transform
        # Skaliranje koordinata u odnosu na 1280x720
        scale_x = width / 1280
        scale_y = height / 720

        src = np.float32([
            [595 * scale_x, 450 * scale_y],   # gore lijevo
            [700 * scale_x, 450 * scale_y],   # gore desno
            [1050 * scale_x, 675 * scale_y],  # dole desno
            [270 * scale_x, 675 * scale_y]    # dole lijevo
        ])

        dst = np.float32([
            [300 * scale_x, 0],
            [980 * scale_x, 0],
            [980 * scale_x, height - 1],
            [300 * scale_x, height - 1]
        ])

        # Transformacija binarne slike
        warped_binary, M, Minv = perspective_transform(
            combined_binary,
            src,
            dst
        )

        # Transformacija na fotografiji u boji
        warped_original, _, _ = perspective_transform(
            undistorted,
            src,
            dst
        )

        cv2.imwrite(
            str(output_dir / f"warped_original_{image_path.stem}.png"),
            warped_original
        )

        # Cuvanje rezultata
        cv2.imwrite(
            str(output_dir / f"warped_{image_path.stem}.png"),
            warped_binary
        )

        # Crtanje trapeza na originalnoj fotografiji
        points_image = undistorted.copy()

        cv2.polylines(
            points_image,
            [src.astype(np.int32)],
            True,
            (0, 255, 0),
            4
        )

        cv2.imwrite(
            str(output_dir / f"perspective_points_{image_path.stem}.png"),
            points_image
        )

        # Pretvaranje binarne slike u BGR radi spajanja
        combined_bgr = cv2.cvtColor(
            combined_binary,
            cv2.COLOR_GRAY2BGR
        )

        name = image_path.stem

        # Cuvanje rezultata
        cv2.imwrite(
            str(output_dir / f"undistorted_{name}.png"),
            undistorted
        )

        cv2.imwrite(
            str(output_dir / f"color_{name}.png"),
            color_binary
        )

        cv2.imwrite(
            str(output_dir / f"gradient_{name}.png"),
            gradient_binary
        )

        cv2.imwrite(
            str(output_dir / f"combined_{name}.png"),
            combined_binary
        )

        print("Processed:", image_path.name)


if __name__ == "__main__":
    main()
