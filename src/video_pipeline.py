import cv2
import numpy as np
from pathlib import Path

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


def process_frame(image, camera_matrix, dist_coeffs, image_size):

    height, width = image.shape[:2]

    # Prilagodjavanje kalibracije dimenzijama kadra
    calib_width, calib_height = image_size

    scale_x = width / calib_width
    scale_y = height / calib_height

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

    combined_binary = cv2.bitwise_or(
        color_binary,
        gradient_binary
    )

    # ROI
    roi_vertices = np.array([
        [int(width * 0.40), int(height * 0.55)],
        [int(width * 0.60), int(height * 0.55)],
        [int(width * 0.90), height - 1],
        [int(width * 0.10), height - 1]
    ], dtype=np.int32)

    roi_binary = region_of_interest(
        combined_binary,
        roi_vertices
    )

    # Perspective Transform
    scale_x = width / 1280
    scale_y = height / 720

    src = np.float32([
        [585 * scale_x, 450 * scale_y],
        [710 * scale_x, 450 * scale_y],
        [1085 * scale_x, height - 1],
        [230 * scale_x, height - 1]
    ])

    dst = np.float32([
        [300 * scale_x, 0],
        [980 * scale_x, 0],
        [980 * scale_x, height - 1],
        [300 * scale_x, height - 1]
    ])

    warped_binary, M, Minv = perspective_transform(
        roi_binary,
        src,
        dst
    )

    # Detekcija linija
    (
        out_img,
        nonzerox,
        nonzeroy,
        left_inds,
        right_inds,
        left_fit,
        right_fit
    ) = sliding_window_search(warped_binary)

    # Pretvaranje piksela u metre
    ym_per_pix = 30 / height
    xm_per_pix = 3.7 / (680 * width / 1280)

    # Zakrivljenost sredine trake
    lane_curvature = calculate_lane_center_curvature(
        left_fit,
        right_fit,
        height,
        xm_per_pix,
        ym_per_pix
    )

    # Polozaj vozila
    vehicle_offset = calculate_vehicle_position(
        left_fit,
        right_fit,
        width,
        height,
        xm_per_pix
    )

    # Crtanje detektovane trake
    final_image = draw_lane(
        undistorted,
        warped_binary,
        left_fit,
        right_fit,
        Minv
    )

    # Prikaz rezultata
    curvature_text = f"Lane curvature: {lane_curvature:.2f} m"

    if vehicle_offset > 0:
        position_text = (
            f"Vehicle position: {abs(vehicle_offset):.2f} m right of center"
        )
    elif vehicle_offset < 0:
        position_text = (
            f"Vehicle position: {abs(vehicle_offset):.2f} m left of center"
        )
    else:
        position_text = "Vehicle position: at center"

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

    return final_image



def main():

    # Putanje do ulaznog i izlaznog foldera
    input_dir = PROJECT_DIR / "test_videos"
    output_dir = PROJECT_DIR / "output_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ucitavanje kalibracionih parametara
    calibration_path = PROJECT_DIR / "camera_cal" / "calibration_data.npz"

    if not calibration_path.exists():
        print("Calibration file not found.")
        return

    with np.load(calibration_path) as data:
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        image_size = tuple(map(int, data["image_size"]))

    # Pronalazenje video-fajlova
    video_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in [".mp4", ".avi", ".mov"]
    )

    if not video_paths:
        print("No videos found in test_videos.")
        return

    for video_path in video_paths:

        print(f"\nProcessing video: {video_path.name}")

        # Otvaranje ulaznog videa
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("Could not open:", video_path.name)
            continue

        # Informacije o videu
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            print("Invalid video properties:", video_path.name)
            cap.release()
            continue

        # Kreiranje izlaznog videa
        output_path = output_dir / f"processed_{video_path.stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            print("Could not create output video.")
            cap.release()
            continue

        frame_number = 0
        failed_frames = 0

        while True:

            # Citanje jednog kadra
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            try:
                # Obrada kadra pomocu postojece funkcije
                processed_frame = process_frame(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    image_size
                )

            except (ValueError, TypeError, IndexError, np.linalg.LinAlgError) as error:
                # Ako detekcija ne uspije, sacuvaj originalni kadar
                failed_frames += 1
                processed_frame = frame

                print(
                    f"Detection failed on frame {frame_number}: {error}"
                )

            # Upis kadra u izlazni video
            writer.write(processed_frame)

            # Prikaz napretka na svakih 30 kadrova
            if frame_number % 30 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

        # Oslobadjanje resursa
        cap.release()
        writer.release()

        print(f"Saved: {output_path}")
        print(f"Frames with failed detection: {failed_frames}")


if __name__ == "__main__":
    main()
