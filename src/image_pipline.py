import cv2
from pathlib import Path
import numpy as np

from color_threshold import color_threshold
from gradient_threshold import gradient_threshold
from perspective_transform import perspective_transform
from region_of_interest import region_of_interest
from lane_detection import (
    sliding_window_search,
    calculate_vehicle_position,
    calculate_lane_center_curvature
)
from lane_visualization import draw_lane

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():

    # Load saved camera calibration parameters
    calibration_path = PROJECT_DIR / "camera_cal" / "calibration_data.npz"

    if not calibration_path.exists():
        print("Calibration file not found. Run camera_calibration.py first.")
        return

    with np.load(calibration_path) as data:
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        image_size = tuple(map(int, data["image_size"]))
        rms = float(data["rms"])

    print("Loaded calibration RMS error:", rms)
    print("Calibration image size:", image_size)
    
    test_dir = PROJECT_DIR / "test_images"
    output_dir = PROJECT_DIR / "output_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all test images
    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not load:", image_path.name)
            continue

        # Get the dimensions of the current image
        height, width = image.shape[:2]

        # Get the dimensions of the calibration images
        calib_width, calib_height = image_size

        # Calculate scaling factors
        scale_x = width / calib_width
        scale_y = height / calib_height

        # Scale the camera matrix to match the current image resolution
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[0, 0] *= scale_x
        scaled_camera_matrix[0, 2] *= scale_x
        scaled_camera_matrix[1, 1] *= scale_y
        scaled_camera_matrix[1, 2] *= scale_y

        # Correct lens distortion
        undistorted = cv2.undistort(
            image,
            scaled_camera_matrix,
            dist_coeffs,
            None,
            scaled_camera_matrix
        )

        # Color and Gradient Threshold
        color_binary = color_threshold(undistorted)
        gradient_binary = gradient_threshold(undistorted)

        # Combine the binary threshold results  
        combined_binary = cv2.bitwise_or(
            color_binary,
            gradient_binary
        )

        # Perspective Transform
        # Scale coordinates relative to 1280x720
        scale_x = width / 1280
        scale_y = height / 720

        # Define source points for the perspective transformation
        src = np.float32([
            [585 * scale_x, 450 * scale_y],    # top-left
            [710 * scale_x, 450 * scale_y],    # top-right
            [1085 * scale_x, (height - 1)],    # bottom-right
            [230 * scale_x, (height - 1)]      # bottom-left
        ])

        # Define a trapezoidal region of interest around the road
        roi_vertices = np.array([
            [int(width * 0.40), int(height * 0.55)],  # top-left
            [int(width * 0.60), int(height * 0.55)],  # top-right
            [int(width * 0.90), height - 1],          # bottom-right
            [int(width * 0.10), height - 1]           # bottom-left
        ], dtype=np.int32)

        # Apply the region of interest mask
        roi_binary = region_of_interest(combined_binary, roi_vertices)

        cv2.imwrite(
            str(output_dir / f"roi_{image_path.stem}.png"),
            roi_binary
        )

        # Define destination points for the bird's-eye view
        dst = np.float32([
            [300 * scale_x, 0],
            [980 * scale_x, 0],
            [980 * scale_x, height - 1],
            [300 * scale_x, height - 1]
        ])

        # Apply perspective transformation to the binary image
        warped_binary, M, Minv = perspective_transform(
            roi_binary,
            src,
            dst
        )

        # Apply perspective transformation to the undistorted color image
        warped_original, _, _ = perspective_transform(
            undistorted,
            src,
            dst
        )

        
        # Detect lane pixels and fit second-degree polynomials
        out_img, nonzerox, nonzeroy, left_inds, right_inds, left_fit, right_fit = sliding_window_search(
            warped_binary
        )

        # Save the sliding window visualization for the selected image
        if image_path.stem == "solidWhiteCurve":
            cv2.imwrite(
                str(output_dir / "sliding_solidWhiteCurve.png"),
                out_img
            )

        # Define approximate pixel-to-meter conversion factors
        ym_per_pix = 30 / height
        xm_per_pix = 3.7 / (680 * width / 1280)

        # Calculate the radius of curvature of the lane center
        lane_curvature = calculate_lane_center_curvature(
            left_fit,
            right_fit,
            height,
            xm_per_pix,
            ym_per_pix
        )
        print(f"Lane center curvature: {lane_curvature:.2f} m")

        # Calculate the vehicle position relative to the center of the lane
        vehicle_offset = calculate_vehicle_position(
            left_fit,
            right_fit,
            width,
            height,
            xm_per_pix
        )

        # Display calculated measurements in the terminal
        print(f"\nImage: {image_path.name}")
        print(f"Vehicle offset: {vehicle_offset:.2f} m")

        # Project the detected lane area back onto the original road image
        final_image = draw_lane(
            undistorted,
            warped_binary,
            left_fit,
            right_fit,
            Minv
        )

        # Prepare the lane curvature text
        curvature_text = f"Lane curvature: {lane_curvature:.2f} m"

        # Prepare the vehicle position text
        if vehicle_offset > 0:
            position_text = f"Vehicle position: {abs(vehicle_offset):.2f} m right of center"
        elif vehicle_offset < 0:
            position_text = f"Vehicle position: {abs(vehicle_offset):.2f} m left of center"
        else:
            position_text = "Vehicle position: at center"

        # Display the lane curvature on the final image
        cv2.putText(
            final_image,
            curvature_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Display the vehicle position on the final image
        cv2.putText(
            final_image,
            position_text,
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Save the final lane detection result
        cv2.imwrite(
            str(output_dir / f"final_{image_path.stem}.png"),
            final_image
        )

        # Save the perspective-transformed color image
        cv2.imwrite(
            str(output_dir / f"warped_original_{image_path.stem}.png"),
            warped_original
        )

        # Save the perspective-transformed binary image
        cv2.imwrite(
            str(output_dir / f"warped_{image_path.stem}.png"),
            warped_binary
        )

        # Draw the source trapezoid on the undistorted image
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

        name = image_path.stem

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
