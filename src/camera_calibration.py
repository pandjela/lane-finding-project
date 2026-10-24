import cv2
import numpy as np
from pathlib import Path

# Path to the project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chessboard dimensions (number of internal corners)
CHESSBOARD_SIZE = (9, 6)

def detect_chessboard_corners(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the internal corners of the chessboard
    found, corners = cv2.findChessboardCorners(
         gray, CHESSBOARD_SIZE, None)
     
    if not found:
        return None

    # Refine corner coordinates for greater accuracy
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )

    refined_corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria
    )

    return refined_corners



def collect_calibration_points():

    image_paths = sorted(
        (PROJECT_DIR / "camera_cal").glob("calibration*.jpg")
    )

    #  Define the known 3D coordinates of chessboard corners
    cols, rows = CHESSBOARD_SIZE

    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[
        0:cols, 0:rows
    ].T.reshape(-1, 2)

    all_object_points = []
    all_image_points = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Could not load {path.name}")
            continue

        current_size = (image.shape[1], image.shape[0])

        # Detect chessboard corners in the current image
        corners = detect_chessboard_corners(image)

        if corners is None:
            print(f"{path.name}: corners not detected")
            continue

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(f"Skipping {path.name}: different image size")
            continue

        # Store corresponding 3D object points and 2D image points
        all_object_points.append(object_points.copy())
        all_image_points.append(corners)

        print(f"{path.name}: {len(corners)} corners detected")

    return all_object_points, all_image_points, image_size


def main():

    # Collect 3D object points and 2D image points
    object_points, image_points, image_size = collect_calibration_points()

    print("\nCalibration images successfully processed:", len(image_points))
    print("Image size:", image_size)


    if len(image_points) < 3 or image_size is None:
        print("Not enough images for camera calibration.")
        return

    # Calculate the camera matrix and distortion coefficients
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    print("\nRMS reprojection error:", rms)

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(dist_coeffs)

    # Save the calibration parameters for later use
    calibration_path = PROJECT_DIR / "camera_cal" / "calibration_data.npz"

    np.savez(
        calibration_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms=rms
    )

    print("\nCalibration parameters saved to:", calibration_path)

    # Load a calibration image to verify distortion correction
    image_path = PROJECT_DIR / "camera_cal" / "calibration3.jpg"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image for undistortion.")
        return

    # Correct lens distortion using the calibration parameters
    undistorted_image = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        camera_matrix
    )

    # Save the  undistorted calibration image
    output_path = PROJECT_DIR / "output_images" / "undistorted_calibration3.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), undistorted_image)

    print("\nUndistorted image saved to:", output_path)



if __name__ == "__main__":
    main()
