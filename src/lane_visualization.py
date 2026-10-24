import cv2
import numpy as np


def draw_lane(original_image, binary_warped, left_fit, right_fit, Minv):

    height, width = binary_warped.shape[:2]

    # Create an empty image for drawing the detected lane area
    color_warp = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate vertical positions across the image
    ploty = np.linspace(0, height - 1, height)

    # Calculate the positions of the left and right lane boundaries
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Create points for the left and right lane boundaries
    pts_left = np.array(
        [np.transpose(np.vstack([left_fitx, ploty]))]
    )

    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]
    )

    # Combine the boundary points into a closed polygon
    pts = np.hstack((pts_left, pts_right)).astype(np.int32)

    # Fill the detected lane area with green color
    cv2.fillPoly(color_warp, [pts], (0, 255, 0))

    # Transform the lane polygon back to the original road perspective
    original_height, original_width = original_image.shape[:2]

    newwarp = cv2.warpPerspective(
        color_warp,
        Minv,
        (original_width, original_height)
    )

    # Overlay the green lane polygon onto the undistorted road image
    result = cv2.addWeighted(
        original_image, 1,
        newwarp, 0.3,
        0
    )

    return result
