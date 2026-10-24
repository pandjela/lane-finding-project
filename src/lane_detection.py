import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def calculate_histogram(binary_warped):

    # Extract the bottom half of the warped binary image
    height = binary_warped.shape[0]
    bottom_half = binary_warped[height // 2:, :]

    # Count nonzero pixels in each column
    histogram = np.sum(bottom_half > 0, axis=0)

    return histogram

    
def sliding_window_search(binary_warped):
    histogram = calculate_histogram(binary_warped)

    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Define sliding window parameters
    nwindows = 9
    margin = 100
    minpix = 50

    height = binary_warped.shape[0]
    window_height = height // nwindows

    # Get the coordinates of all nonzero pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Create an image for visualization
    out_img = np.dstack((binary_warped,) * 3)

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the sliding windows
        cv2.rectangle(out_img,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)

        cv2.rectangle(out_img,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        # Find nonzero pixels inside each window
        good_left_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &
            (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &
            (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter the windows based on the mean position of detected pixels
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Highlight detected left lane pixels in red
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]

    # Highlight detected right lane pixels in blue
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    
    # Get the coordinates of detected left lane pixels
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    # Get the coordinates of detected right lane pixels
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

     # Check whether enough pixels were detected for polynomial fitting
    if len(leftx) < 3 or len(rightx) < 3:
        raise ValueError("Not enough pixels for polynomial fitting.")

    # Fit second-degree polynomials to both lane boundaries
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
   
    # Generate vertical positions across the image
    ploty = np.linspace(0, height - 1, height)

    # Calculate horizontal positions using the fitted polynomials
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Create points for drawing the polynomial curves
    left_points = np.column_stack((left_fitx, ploty)).astype(np.int32)
    right_points = np.column_stack((right_fitx, ploty)).astype(np.int32)

    # Draw the fitted polynomial curves in yellow
    cv2.polylines(out_img, [left_points], False, (0, 255, 255), 4)
    cv2.polylines(out_img, [right_points], False, (0, 255, 255), 4)

    print("Left polynomial coefficients:", left_fit)
    print("Right polynomial coefficients:", right_fit)


    return out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit



def calculate_curvature(leftx, lefty, rightx, righty, image_height,
                        xm_per_pix, ym_per_pix):

    # Fit second-degree polynomials in real-world coordinates (meters)
    left_fit_m = np.polyfit(
        lefty * ym_per_pix,
        leftx * xm_per_pix,
        2
    )

    right_fit_m = np.polyfit(
        righty * ym_per_pix,
        rightx * xm_per_pix,
        2
    )

    # Evaluate curvature at the bottom of the image
    y_eval = (image_height - 1) * ym_per_pix

    def curvature_radius(fit):
        A, B = fit[0], fit[1]

        if abs(A) < 1e-12:
            return float("inf")

        return ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / abs(2 * A)

    left_curvature = curvature_radius(left_fit_m)
    right_curvature = curvature_radius(right_fit_m)

    return left_curvature, right_curvature



def calculate_lane_center_curvature(
    left_fit, right_fit, image_height, xm_per_pix, ym_per_pix
):
    # Generate vertical positions across the image
    ploty = np.linspace(0, image_height - 1, image_height)

    # Calculate the positions of the left and right lane boundaries
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # Calculate the center of the lane
    center_fitx = (left_fitx + right_fitx) / 2

    # Fit a second-degree polynomial to the lane center in meters
    center_fit_m = np.polyfit(
        ploty * ym_per_pix,
        center_fitx * xm_per_pix,
        2
    )

    # Evaluate curvature at the bottom of the image
    y_eval = (image_height - 1) * ym_per_pix

    A = center_fit_m[0]
    B = center_fit_m[1]

    if abs(A) < 1e-12:
        return float("inf")

    curvature = (
        (1 + (2 * A * y_eval + B) ** 2) ** 1.5
    ) / abs(2 * A)

    return curvature



def calculate_vehicle_position(left_fit, right_fit, image_width,
                               image_height, xm_per_pix):

    # Evaluate lane positions at the bottom of the image
    y_eval = image_height - 1

    # Calculate the positions of both lane boundaries
    left_x = np.polyval(left_fit, y_eval)
    right_x = np.polyval(right_fit, y_eval)

    # Calculate the center of the lane
    lane_center = (left_x + right_x) / 2

    # Assume the camera is positioned at the center of the vehicle
    vehicle_center = image_width / 2

    # Calculate the vehicle offset from the lane center in meters
    offset = (vehicle_center - lane_center) * xm_per_pix

    return offset




def main():

    image_path = PROJECT_DIR / "output_images" / "warped_test1.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not load image.")
        return

   
    out_img, nonzerox, nonzeroy, left_inds, right_inds, left_fit, right_fit = sliding_window_search(image)


    # Get the coordinates of detected left lane pixels
    leftx = nonzerox[left_inds]
    lefty = nonzeroy[left_inds]

    # Get the coordinates of detected right lane pixels
    rightx = nonzerox[right_inds]
    righty = nonzeroy[right_inds]

     # Get image dimensions
    height, width = image.shape

    # Define approximate pixel-to-meter conversion factors
    ym_per_pix = 30 / height
    xm_per_pix = 3.7 / (680 * width / 1280)

    # Calculate the radius of curvature for both lane boundaries
    left_curvature, right_curvature = calculate_curvature(
        leftx, lefty,
        rightx, righty,
        height,
        xm_per_pix,
        ym_per_pix
    )

    print(f"Left lane curvature: {left_curvature:.2f} m")
    print(f"Right lane curvature: {right_curvature:.2f} m")

    
    # Calculate the vehicle position relative to the lane center
    vehicle_offset = calculate_vehicle_position(
        left_fit,
        right_fit,
        width,
        height,
        xm_per_pix
    )

    if vehicle_offset > 0:
        print(f"Vehicle is {abs(vehicle_offset):.2f} m right of center")
    elif vehicle_offset < 0:
        print(f"Vehicle is {abs(vehicle_offset):.2f} m left of center")
    else:
        print("Vehicle is at the center of the lane")

    
    
    cv2.imwrite(    
    str(PROJECT_DIR / "output_images" / "curvature_test1.png"),
    out_img)
    

    histogram = calculate_histogram(image)

    # Find the midpoint of the histogram
    midpoint = len(histogram) // 2

    # Find the peak in the left half
    leftx_base = np.argmax(histogram[:midpoint])

    # Find the peak in the right half
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    print("Left lane base:", leftx_base)
    print("Right lane base:", rightx_base)

    plt.figure()
    plt.title("Lane Pixel Histogram")
    plt.xlabel("Horizontal pixel position (x)")
    plt.ylabel("Number of white pixels")
    plt.plot(histogram)
    plt.xlim([0, image.shape[1]])

    plt.show()


if __name__ == "__main__":
    main()
