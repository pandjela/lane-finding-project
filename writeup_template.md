## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The first step of the lane detection pipeline was camera calibration. The purpose of this step was to determine the camera matrix and distortion coefficients, which are used to correct lens distortion in the input images.

Camera calibration was performed using a set of chessboard images stored in the `camera_cal` directory. The chessboard pattern contained 9 × 6 internal corners. OpenCV's `cv2.findChessboardCorners()` function was used to detect the corner positions, while `cv2.cornerSubPix()` was used to refine their coordinates for greater accuracy. The detected image points were matched with their corresponding 3D object points on the chessboard plane.

A total of 20 calibration images were available, and 15 images were used for calibration. Images where the corners could not be detected or whose dimensions differed from the accepted calibration image size were skipped.

The collected image points and object points were then passed to `cv2.calibrateCamera()` to calculate the camera matrix and distortion coefficients.

The resulting RMS reprojection error was approximately **0.853 pixels**. The calibration parameters were saved in `camera_cal/calibration_data.npz` so they could be reused without repeating the calibration process.

Finally, `cv2.undistort()` was used to correct lens distortion. The correction was first tested on a calibration image and later applied to road images in the main processing pipeline. Since the calibration images had a resolution of 1280 × 720 pixels, the camera matrix was scaled when processing road images with different resolutions.

The following images show an example of a calibration image before and after distortion correction.

**Original calibration image:**

![Original calibration image](./camera_cal/calibration3.jpg)

**Undistorted calibration image:**

![Undistorted calibration image](./output_images/undistorted_calibration3.jpg)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction was applied to the road images in the main processing pipeline, implemented in src/image_pipline.py. Each image was corrected using the previously calculated camera matrix and distortion coefficients before applying further image processing techniques.

**Original road image:**

![Original calibration image](./test_images/test1.jpg)

**Undistorted road image:**

![Undistorted calibration image](./output_images/undistorted_test1.png)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create thresholded binary image, I combined color-based thresholding and gradient-based thresholding. These methods were implemented separately in src/color_threshold.py and src/gradient_threshold.py, and their results were combined in the main processing pipeline in src/image_pipline.py.

**Color Thresholding**
First, I converted the undistorted image from BGR to HLS color space using cv2.cvtColor(). The HLS color space was used to help identify white and yellow lane markings.
I defined separate color ranges for white and yellow lane lines and created binary masks using cv2.inRange(). The two masks were then combined using cv2.bitwise_or() to obtain a single color-based binary image.

**Gradient Thresholding**

For gradient-based thresholding, I first converted the undistorted image to grayscale. Then, I applied the Sobel operator in the X direction using cv2.Sobel() to detect changes in pixel intensity, particularly along the edges of lane markings.

The absolute gradient values were scaled to the range 0–255, and a threshold was applied to create a binary image. The final implementation uses a Sobel kernel size of 3 and a threshold range of 25–120.

I also tested different Sobel kernel sizes (3 and 5) and threshold ranges (25–120 and 20–180) to compare their results.

**Combining the Results**

Finally, I combined the color and gradient binary images using cv2.bitwise_or(). This allowed pixels detected by either method to be included in the final binary image.

The combined method successfully highlighted the lane markings in the test images. However, some additional objects, such as trees, vehicles, and road signs, were also detected because of their colors or strong edges.

The resulting binary images were then used in the next stage of the pipeline, where a region of interest was applied before the perspective transformation.

Example results:

**Original road image:**

![Original road image](./test_images/solidWhiteCurve.jpg)

**Color threshold result:**

![Color threshold image](./output_images/color_solidWhiteCurve.png)

**Gradient threshold result:**

![Gradient threshold image](./output_images/gradient_solidWhiteCurve.png)

**Combined binary result:**

![Combined binary result image](./output_images/combined_solidWhiteCurve.png)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To obtain a bird's-eye view of the road, I applied a perspective transform to the combined binary image. The transformation was implemented in src/perspective_transform.py and integrated into the main processing pipeline in src/image_pipline.py.

Before applying the perspective transformation, I defined a trapezoidal region of interest to focus on the road area where lane markings were expected. The ROI was applied to the combined binary image using a mask created with cv2.fillPoly() and cv2.bitwise_and().
This step removed pixels outside the selected region, reducing the influence of irrelevant objects and image features.

First, I manually selected four source points (src) defining a trapezoidal region around the lane boundaries. I also defined four destination points (dst) forming a rectangle in the transformed image. The source and destination coordinates were scaled according to the image dimensions to support test images with different resolutions.

I used cv2.getPerspectiveTransform() to calculate the transformation matrix M, which maps the source points to the destination points. I also calculated the inverse transformation matrix Minv, which will later be used to project the detected lane area back onto the original road image.

The perspective transformation was applied using cv2.warpPerspective(). For binary images, I used nearest-neighbor  (cv2.INTER_NEAREST) interpolation to preserve the binary pixel values.

I tested the transformation on several road images, including straight and curved roads. Some differences remained on curved roads, where the lane markings were not always perfectly aligned.

The transformed binary images will be used in the next stage of the pipeline to detect lane-line pixels and fit polynomial curves.

**Original road image solidYellowCurve.jpg:**

![Original road image](./test_images/solidYellowCurve.jpg)

**Selected source points for the perspective transformation:**

![Perspective transformation source points](./output_images/perspective_points_solidYellowCurve.png)

**Perspective-transformed road image:**

![Perspective-transformed road image](./output_images/warped_original_solidYellowCurve.png)

**Perspective-transformed binary image:**

![Perspective-transformed binary imagee](./output_images/warped_solidYellowCurve.png)


-------------------------------------------------------------------------------------------------------------------------
**Original road image test1.jpg:**

![Original road image](./test_images/test1.jpg)

**Selected source points for the perspective transformation:**

![Perspective transformation source points](./output_images/perspective_points_test1.png)

**Perspective-transformed road image:**

![Perspective-transformed road image](./output_images/warped_original_test1.png)

**Perspective-transformed binary image:**

![Perspective-transformed binary imagee](./output_images/warped_test1.png)


-------------------------------------------------------------------------------------------------------------------------

**Original road image test2.jpg:**

![Original road image](./test_images/test2.jpg)

**Selected source points for the perspective transformation:**

![Perspective transformation source points](./output_images/perspective_points_test2.png)

**Perspective-transformed road image:**

![Perspective-transformed road image](./output_images/warped_original_test2.png)

**Perspective-transformed binary image:**

![Perspective-transformed binary imagee](./output_images/warped_test2.png)


-------------------------------------------------------------------------------------------------------------------------

**Original road image challange00101.jpg:**

![Original road image](./test_images/challange00101.jpg)

**Selected source points for the perspective transformation:**

![Perspective transformation source points](./output_images/perspective_points_challange00101.png)

**Perspective-transformed road image:**

![Perspective-transformed road image](./output_images/warped_original_challange00101.png)

**Perspective-transformed binary image:**

![Perspective-transformed binary imagee](./output_images/warped_challange00101.png)

-------------------------------------------------------------------------------------------------------------------------

**Original road image straight_lines2.jpg:**

![Original road image](./test_images/straight_lines2.jpg)

**Selected source points for the perspective transformation:**

![Perspective transformation source points](./output_images/perspective_points_straight_lines2.png)

**Perspective-transformed road image:**

![Perspective-transformed road image](./output_images/warped_original_straight_lines2.png)

**Perspective-transformed binary image:**

![Perspective-transformed binary imagee](./output_images/warped_straight_lines2.png)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect lane-line pixels, I used a histogram-based Sliding Windows algorithm. This step was implemented in `src/lane_detection.py`, using the perspective-transformed binary images from the previous stage of the pipeline.

First, I calculated a histogram of the lower half of the binary image by counting the nonzero pixels in each column using `np.sum()`. I divided the histogram into two halves and used `np.argmax()` to find the positions with the highest number of detected pixels. These positions were used as the starting points for the left and right lane lines.

Next, I applied the Sliding Windows algorithm to follow the lane markings from the bottom to the top of the image. I divided the image into 9 horizontal windows, with a margin of 100 pixels on each side. For each window, I identified the nonzero pixels within its boundaries. If more than 50 pixels were detected, the horizontal position of the next window was updated using the mean x-coordinate of those pixels.

After collecting the pixels belonging to the left and right lane lines, I checked whether at least three pixels were detected for each line. If there were not enough pixels, the algorithm raised an error instead of attempting polynomial fitting.

I then used np.polyfit() to fit a second-degree polynomial to each line. The polynomial describes the horizontal position of the lane boundary as a function of the vertical image coordinate:

$$
x(y) = Ay^2 + By + C
$$

I used np.polyval() to calculate the lane positions along the image height and cv2.polylines() to draw the fitted curves. The detected pixels were also highlighted in different colors to visualize the Sliding Windows results.

I tested this method on several images, including test1.jpg, test2.jpg, and solidYellowCurve.jpg. The Sliding Windows algorithm successfully followed both lane boundaries in the successful examples, including dashed lane markings. The fitted polynomials generally matched the detected lane pixels, although some deviations were visible, especially where fewer lane pixels were available.

**Sliding Windows and polynomial fitting results:**

**Lane detection on test1.jpg:**

![Polynomial fitting test1](./output_images/polynomial2_test1.png)

**Lane detection on test2.jpg:**

![Polynomial fitting test2](./output_images/polynomial2_test2.png)

**Lane detection on solidYellowCurve.jpg:**

![Polynomial fitting curved road](./output_images/polynomial2_solidYellowCurve.png)

**Lane detection on solidWhiteCurve.jpg:**

![Polynomial fitting curved road](./output_images/polynomial2_solidWhiteCurve.png)

**Lane detection on challange00101.jpg:**

![Polynomial fitting curved road](./output_images/polynomial2_challange00101.png)



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After detecting the lane boundaries and fitting second-degree polynomials, I calculated the radius of curvature of the lane and the vehicle's position relative to the lane center. These calculations were implemented in src/lane_detection.py, using the functions calculate_lane_center_curvature() and calculate_vehicle_position(). Both functions were called from the main processing pipeline in src/image_pipline.py.

Radius of Curvature

To calculate the lane curvature, I first determined the horizontal positions of the left and right lane boundaries using their fitted polynomials. I then calculated the center of the lane by averaging the corresponding positions of both boundaries.

Since the polynomial coordinates were initially expressed in pixels, I used approximate conversion factors to convert them into meters:

ym_per_pix = 30 / image_height for the vertical direction.

xm_per_pix = 3.7 / (680 * image_width / 1280) for the horizontal direction.

These values were based on the assumptions that the visible road section represents approximately 30 meters and the lane width is approximately 3.7 meters.

I then used np.polyfit() to fit a second-degree polynomial to the lane center in real-world coordinates.

The radius of curvature was calculated using the following formula:

$$
R = \frac{(1 + (2Ay + B)^2)^{3/2}}{|2A|}
$$

where A and B are the polynomial coefficients, and y is the vertical position at which the curvature is evaluated.

The curvature was evaluated at the bottom of the image, representing the road area closest to the vehicle. The result was expressed in meters.

Vehicle Position Relative to the Lane Center

To determine the vehicle's position, I calculated the horizontal positions of the left and right lane boundaries at the bottom of the image using np.polyval().

The lane center was calculated as the average of these two positions. I assumed that the camera was mounted at the horizontal center of the vehicle, so the vehicle center corresponded to half of the image width.
The lane center was calculated using:

$$
x_l = \frac{x_{left} + x_{right}}{2}
$$

The vehicle offset was calculated using:

$$
d = (x_v - x_l) \cdot s_x
$$

where:
- d is the vehicle offset in meters.
- x_v is the horizontal position of the vehicle center.
- x_l is the horizontal position of the lane center.
- s_x is the pixel-to-meter conversion factor (xm_per_pix).

A positive offset indicated that the vehicle was positioned to the right of the lane center, while a negative offset indicated that it was positioned to the left.

Both the calculated radius of curvature and the vehicle offset were later displayed on the final processed road image.

These measurements are approximate because they depend on the assumed pixel-to-meter conversion factors and the accuracy of the detected lane boundaries.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After detecting the lane boundaries and calculating the lane curvature and vehicle position, I projected the detected lane area back onto the original road perspective. This step was implemented in src/lane_visualization.py using the draw_lane() function and integrated into the main processing pipeline in src/image_pipline.py.

First, I used np.polyval() to calculate the horizontal positions of the left and right lane boundaries along the height of the perspective-transformed image. These points were combined to create a polygon representing the area between the detected lane lines.

I then used cv2.fillPoly() to fill this polygon with green color on a separate image.

To return the detected lane area to the original road perspective, I applied cv2.warpPerspective() using the inverse perspective transformation matrix (Minv) calculated earlier in the pipeline.

The resulting green lane area was overlaid onto the undistorted road image using cv2.addWeighted(), with a blending factor of 0.3 to keep the road visible underneath.

Finally, I used cv2.putText() to display the calculated lane curvature and vehicle position relative to the lane center on the resulting image.

The final output shows the detected lane area highlighted in green, together with the estimated radius of curvature and vehicle offset. This makes it easier to visualize the detected lane boundaries and evaluate the results of the complete processing pipeline.

**Example of the final lane detection result:**

![Identified lane area on the road](./output_images/final_test1.png)

![Identified lane area on the road](./output_images/final_test2.png)

![Identified lane area on the road](./output_images/final_test5.png)

![Identified lane area on the road](./output_images/final_solidWhiteCurve.png)

![Identified lane area on the road](./output_images/final_solidYellowCurve.png)

![Identified lane area on the road](./output_images/final_straight_lines2.png)

![Identified lane area on the road](./output_images/final_challange00101.png)


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

After implementing and testing the lane detection pipeline on individual road images, I extended it to process video files. The video processing functionality was implemented in src/video_pipeline.py. The main idea was to apply the same lane detection steps to each frame of the video and combine the processed frames into a new output video.

The process_frame() function performs the following operations on each frame:

    -Adjusts the camera calibration matrix to match the frame resolution and corrects lens distortion using cv2.undistort().

    -Applies color and gradient thresholding to identify potential lane markings.

    -Combines the binary threshold results and applies a region of interest to focus on the road.

    -Performs a perspective transformation to obtain a bird's-eye view.

    -Detects lane pixels using the Sliding Windows algorithm and fits second-degree polynomials to the left and right lane boundaries.

    -Calculates the lane curvature and the vehicle's position relative to the lane center.

    -Projects the detected lane area back onto the original road perspective and highlights it in green.

    -Displays the calculated lane curvature and vehicle offset on the processed frame.

The main() function reads video files from the test_videos directory using cv2.VideoCapture(). Each frame is processed independently by calling process_frame(). The processed frames are then written to a new video using cv2.VideoWriter(), preserving the input video's frame dimensions and frame rate.
The output videos are saved in the output_videos directory with the prefix processed_.

I also added exception handling for cases where lane detection fails on a particular frame. If an error occurs during processing, the original frame is written to the output video instead. This allows processing to continue without stopping the entire video because of a single failed frame. The program also counts and reports the number of frames where detection failed.

The video pipeline uses the same detection parameters for each frame and does not use information from previous frames. As a result, changes in lighting, road markings, or surrounding objects can cause the detected lane boundaries to vary between consecutive frames.

Video processing results:

The pipeline was tested on multiple video files containing different road conditions. The results demonstrate that the algorithm can detect and visualize lane boundaries across consecutive frames, although detection accuracy varies depending on the road scene.

The processed videos are available in the output_videos directory.

**Final video output:**

![Video output](./output_videos/processed_challenge01.mp4)

![Video output](./output_videos/processed_project_video01.mp4)

![Video output](./output_videos/processed_project_video02.mp4)

![Video output](./output_videos/processed_project_video03.mp4)

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the implementation of this project, I encountered several challenges related to camera calibration, lane pixel detection, perspective transformation, and processing different road conditions.

One of the initial challenges was camera calibration. Not all chessboard images could be used successfully because the required corner pattern was not always detected, and some images had different resolutions. To address this, I used only the images with successfully detected corners and matching dimensions. I also adjusted the camera matrix when processing road images with different resolutions.

Another challenge was selecting appropriate color and gradient thresholds. Although the combination of HLS color thresholding and Sobel gradient detection worked well on many images, it sometimes detected additional objects such as road signs, vehicles, and surrounding vegetation. Applying a region of interest helped reduce these unwanted detections, but it did not eliminate them completely.

Perspective transformation was also challenging because the source and destination points had to be selected manually. The selected coordinates worked reasonably well for several test images, but they were not equally suitable for every road scene, particularly on curved roads or when the road geometry differed significantly.

During lane detection, the Sliding Windows algorithm generally identified the left and right lane boundaries successfully when the road markings were clearly visible. However, the results were less reliable when lane markings were faded, partially missing, or surrounded by other strong edges. In these situations, the detected pixels could produce inaccurate polynomial fits, affecting the estimated lane curvature and vehicle position.

The video pipeline introduced an additional challenge. The algorithm processes each video frame independently, without using the polynomial coefficients detected in previous frames. As a result, the detected lane boundaries may change slightly between consecutive frames, causing the green lane overlay to move or become unstable. Some test videos produced consistent results, while others showed less accurate lane detection.

To prevent individual detection errors from interrupting video processing, I added exception handling. When lane detection fails on a frame, the original frame is written to the output video, allowing the remaining frames to be processed.

**Possible improvements**

The pipeline could be made more robust by improving the thresholding methods to handle different lighting conditions, shadows, and road surfaces. The perspective transformation could also be adapted more dynamically instead of relying on fixed source and destination points.

Another improvement would be to use previously detected lane polynomials to search for lane pixels in subsequent video frames. This could reduce repeated calculations and help maintain more consistent lane tracking. Temporal smoothing could also be applied to reduce sudden changes in the detected lane boundaries and curvature estimates.

Additionally, validation checks could be introduced to verify that the detected lane boundaries have reasonable curvature, separation, and relative positions before accepting the results.

Finally, the calculations of lane curvature and vehicle position currently rely on approximate pixel-to-meter conversion factors. More accurate camera and road measurements could improve the reliability of these estimates.

Overall, the project demonstrates a complete lane detection pipeline, from camera calibration and image preprocessing to lane boundary detection, curvature estimation, and visualization. The implementation works reasonably well on several test images and videos, but its performance depends on the visibility of lane markings and the road conditions. The proposed improvements could make the system more stable and reliable in more challenging driving scenarios.

