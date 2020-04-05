import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Globals
mtx = None
dist = None

# Calibrate camera API
def calibrate_camera(display_output = False):
    # Chess board size
    chessboard_size = (9, 6)

    # Since we know object points, we can prepare them as (0, 0, 0), (1, 0, 0) ...
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Prepare input arrays for cv2.calibrateCamera()
    object_points = []
    image_points = []

    # Load all images from camera_cal folder
    images = glob.glob('camera_cal/calibration*.jpg')

    image_shape = None

    # Iterate through images and append image points for coresponding
    for image in images:
        # Read image
        img = cv2.imread(image)

        image_shape = img.shape

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # Check if found corners successfuly
        if ret is True:
            # Append detected corners alongisde coresponding objp
            object_points.append(objp)
            image_points.append(corners)

            # Display found corners as sanity check
            if display_output is True:
                cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                cv2.imshow('Corners', img)
                cv2.waitKey(200)
                cv2.destroyAllWindows()
        else:
            # Opencv findChessboardCorners fails for for calibration images 1, 4, 5
            # I guess the reason is missing whitespace around chessboard in those images
            
            # Note from opencv site:

            '''
            The function requires white space (like a square-thick border, the wider the better) around the board to make the detection more robust in various 
            environments. Otherwise, if there is no border and the background is dark, the outer black squares cannot be segmented properly and so the square 
            grouping and ordering algorithm fails.
            '''
            print("Failed to find chessbpard corners for", image)

    # Acquire camera matrix and distortion coeffs
    ret, mtx, dist_coef, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (image_shape[1], image_shape[0]), None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist_coef
    pickle.dump(dist_pickle, open("calibration_output/wide_dist_pickle.p", "wb"))

# Image processing pipeline API
def undistort_image(image, camera_matrix, distortion_coefficients):
    # Just apply opencv distortion
    return cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)

def absolute_sobel_thresholding(img, thresh_low = 20, thresh_high = 100, kernel_size = 5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply sobel in x direction
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = kernel_size)

    # Absolute value
    absolute_sobel = np.abs(sobel)

    # Scale absolute to 0 - 255 range
    scaled_absolute = np.uint8(255 * absolute_sobel / np.max(absolute_sobel))

    # Prepare output
    binary_output = np.zeros_like(scaled_absolute)

    # Calculate output
    binary_output[(scaled_absolute > thresh_low) & (scaled_absolute < thresh_high)] = 1

    return binary_output

def color_thresholding(img, thresh_low = 170, thresh_high = 255):
    # Convert image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Isolate S channel
    s_channel = hls[:, :, 2]

    # Calculate binary output
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh_low) & (s_channel <= thresh_high)] = 1

    return binary_output

def gradient_direction_thresholding(img, sobel_kernel=5, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get gradient in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)

    # Get gradient in y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Get absolute values for each image
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)

    # Calculate gradient direction
    gradient_direction = np.arctan2(abs_sobel_y, abs_sobel_x)

    # Prepare output
    binary_output = np.zeros_like(gradient_direction)

    # Do the thresholding
    l_thresh, h_thresh = thresh
    binary_output[(gradient_direction >= l_thresh) & (gradient_direction <= h_thresh)] = 1

    return binary_output

# Will not be used since it introduces additional y direction edges
def gradient_magnitude_thresholding(img, sobel_kernel=5, mag_thresh=(0, 255)):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get gradient in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)

    # Get gradient in y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Calculate magnitude as mag = sqrt(sobel_x^2 + sobel_y^2)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Scale the magnitude to 0 - 255 range
    scaled_magniuted = np.uint8(255 * magnitude / np.max(magnitude))

    # Create binary mask
    binary_output = np.zeros_like(scaled_magniuted)

    low_thresh, high_thresh = mag_thresh

    # Do the thresholding
    binary_output[(low_thresh <= scaled_magniuted) & (scaled_magniuted <= high_thresh)] = 1

    return binary_output

def apply_thresholds(img,
     blur_kernel_size = 5, 
     abs_sobel_kernel = 5,
     abs_soble_threshold = (20, 100),
     color_thresholds = (170, 255), 
     gradient_magnitude_kernel_size = 5, 
     gradient_magnitude_thresholds = (80, 100), 
     gradient_kernel_size = 15, 
     gradient_direction_thresholds = (0.7, 1.4)):

    # Apply bluring to the image
    img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

    # Sobel gradient thresholding
    gradient_binary = absolute_sobel_thresholding(img, thresh_low = abs_soble_threshold[0], thresh_high = abs_soble_threshold[1], kernel_size = abs_sobel_kernel)

    # S channel thresholding
    color_binary = color_thresholding(img, thresh_low = color_thresholds[0], thresh_high = color_thresholds[1])

    # Apply gradient magnitude thresholding
    mag_binary = gradient_magnitude_thresholding(img, sobel_kernel = gradient_magnitude_kernel_size, mag_thresh = gradient_magnitude_thresholds)

    # Apply gradient direction thresholding
    direction_binary = gradient_direction_thresholding(img, sobel_kernel = gradient_kernel_size, thresh = gradient_direction_thresholds)

    # Prepare binary output
    combined_binary = np.zeros_like(gradient_binary)

    # Combine thresholds
    combined_binary[((mag_binary == 1) & (direction_binary == 1)) | (color_binary == 1) | (gradient_binary == 1)] = 1

    return combined_binary

def go_to_birdview_perspective(img):
    # Start by defining source points
    # Magic numbers acquired from gimp
    point1 = [280, 700]
    point2 = [595, 460]
    point3 = [725, 460]
    point4 = [1125, 700]
    source = np.float32([point1, point2, point3, point4])

    # Define destination vertices
    # Magic numbers acquired from gimp
    dest_point1 = [250, 720]
    dest_point2 = [250, 0]
    dest_point3 = [1065, 0]
    dest_point4 = [1065, 720]
    destination = np.float32([dest_point1, dest_point2, dest_point3, dest_point4])

    transformation_matrix = cv2.getPerspectiveTransform(source, destination)
    inverse_transform_matrix = cv2.getPerspectiveTransform(destination, source)

    output = cv2.warpPerspective(img, transformation_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return output, inverse_transform_matrix

def find_lane_start(binary_warped, middle_offset = 100, corner_offset = 100):
    # Sum all the ones in binary image per column
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis = 0)

    # Find the mid point in histogram
    midpoint = np.int(histogram.shape[0] // 2)
    
    # Find left peak
    left_base = np.argmax(histogram[0 + corner_offset : midpoint - middle_offset])

    # Find right peak
    right_base = np.argmax(histogram[midpoint + middle_offset : histogram.shape[0] - corner_offset]) + midpoint

    return left_base, right_base

def find_lane_pixels(binary_warped, number_of_windows = 9, margin = 100, min_pixel_detection = 50, draw_windows = False):
    # Calculate height of single window
    window_height = np.int(binary_warped.shape[0] // number_of_windows)

    # Find indices of all non zero pixels in the image
    non_zero = binary_warped.nonzero()
    non_zero_x = np.array(non_zero[1])
    non_zero_y = np.array(non_zero[0])

    # Set position of current window
    left_current_x, right_current_x = find_lane_start(binary_warped)

    # Prepare output lists in which we put indices of pixels that belong to the lane line
    left_lane_indices = []
    right_lane_indices = []

    # Optional draw output
    if draw_windows is True:
        out_image = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    for window in range(number_of_windows):
        # Identify window vertical boundaries
        window_y_low = binary_warped.shape[0] - (window + 1) * window_height
        window_y_high = binary_warped.shape[0] - window * window_height

        # Identify window horizontal boundaries
        left_window_x_low = left_current_x - margin
        left_window_x_high = left_current_x + margin

        right_window_x_low = right_current_x - margin
        right_window_x_high = right_current_x + margin

        # Optional draw
        if draw_windows is True:
            cv2.rectangle(out_image,(left_window_x_low, window_y_low),
                (left_window_x_high, window_y_high), (0,255,0), 4)
            cv2.rectangle(out_image,(right_window_x_low, window_y_low),
                (right_window_x_high, window_y_high),(0,255,0), 4)
        
        # Identify all pixels that belong to left window
        left_belonging_pixels_indices = ((non_zero_y >= window_y_low) &
            (non_zero_y < window_y_high) &
            (non_zero_x >= left_window_x_low) &
            (non_zero_x < left_window_x_high)).nonzero()[0]

        # Identify all pixels that belong to right window
        right_belonging_pixels_indices = ((non_zero_y >= window_y_low) &
            (non_zero_y < window_y_high) &
            (non_zero_x >= right_window_x_low) &
            (non_zero_x < right_window_x_high)).nonzero()[0]

        # Record belonging left line indices
        left_lane_indices.append(left_belonging_pixels_indices)

        # Record belonging right line indices
        right_lane_indices.append(right_belonging_pixels_indices)

        # Recalculate center of next iteration window
        if len(left_belonging_pixels_indices) > min_pixel_detection:
            left_current_x = int(np.mean(non_zero_x[left_belonging_pixels_indices]))

        if len(right_belonging_pixels_indices) > min_pixel_detection:
            right_current_x = int(np.mean(non_zero_x[right_belonging_pixels_indices]))

    # Concatenate all the recorded pixel coordinates
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right lane line pixel positions
    left_x = non_zero_x[left_lane_indices]
    left_y = non_zero_y[left_lane_indices]

    right_x = non_zero_x[right_lane_indices]
    right_y = non_zero_y[right_lane_indices]

    if draw_windows is True:
        return left_x, left_y, right_x, right_y, out_image
    else:
        return left_x, left_y, right_x, right_y

def fit_poly(line_x, line_y):
    poly = np.polyfit(line_y, line_x, 2)
    return poly

def fit_polynomial(left_line_x, left_line_y, right_line_x, right_line_y, img_shape, draw_lines = False, out_img = None):
    # Fit poly
    # Reverse x and y because we for single x value, we can have multiple points
    left_line = np.polyfit(left_line_y, left_line_x, 2)
    right_line = np.polyfit(right_line_y, right_line_x, 2)

    if draw_lines is not True:
        return left_line, right_line
    else:
        # Calculate concrete values so we can plot line
        plot_y = np.linspace(0, img_shape[0] - 1, img_shape[0])
        left_line_ploted = left_line[0] * plot_y ** 2 + left_line[1] * plot_y + left_line[2]
        right_line_ploted = right_line[0] * plot_y ** 2 + right_line[1] * plot_y + right_line[2]
        
        # Prepare output image
        out_img[left_line_y, left_line_x] = [255, 0, 0]
        out_img[right_line_y, right_line_x] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_line_ploted, plot_y, color='yellow')
        plt.plot(right_line_ploted, plot_y, color='yellow')

        return left_line, right_line, out_img

def targeted_search(warped_binary, prev_left_line, prev_right_line, margin = 100, draw_lines = False):
    # Find non zero pixels
    non_zero = warped_binary.nonzero()
    non_zero_x = non_zero[1]
    non_zero_y = non_zero[0]

    # Search for points around previous lane detection, similar to what we did with windows
    left_belonging_pixel_indices = ((non_zero_x > (prev_left_line[0] * (non_zero_y ** 2) + prev_left_line[1] * non_zero_y + prev_left_line[2] - margin)) &
        (non_zero_x < (prev_left_line[0] * (non_zero_y ** 2) + prev_left_line[1] * non_zero_y + prev_left_line[2] + margin)))

    right_belonging_pixel_indices = ((non_zero_x > (prev_right_line[0] * (non_zero_y ** 2) + prev_right_line[1] * non_zero_y + prev_right_line[2] - margin)) &
        (non_zero_x < (prev_right_line[0] * (non_zero_y ** 2) + prev_right_line[1] * non_zero_y + prev_right_line[2] + margin)))

    # Extract left and right lane line pixel positions
    left_line_x = non_zero_x[left_belonging_pixel_indices]
    left_line_y = non_zero_y[left_belonging_pixel_indices]

    right_line_x = non_zero_x[right_belonging_pixel_indices]
    right_line_y = non_zero_y[right_belonging_pixel_indices]

    if draw_lines is False:
        return left_line_x, left_line_y, right_line_x, right_line_y
    else:
        left_line = fit_poly(left_line_x, left_line_y)
        right_line = fit_poly(right_line_x, right_line_y)

        # Calculate concrete values so we can plot line
        plot_y = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
        left_line_ploted = left_line[0] * plot_y ** 2 + left_line[1] * plot_y + left_line[2]
        right_line_ploted = right_line[0] * plot_y ** 2 + right_line[1] * plot_y + right_line[2]

        # Prepare output image
        out_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255
        out_img[left_line_y, left_line_x] = [255, 0, 0]
        out_img[right_line_y, right_line_x] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_line_ploted, plot_y, color='white')
        plt.plot(right_line_ploted, plot_y, color='white')

        return left_line_x, left_line_y, right_line_x, right_line_y, out_img

def measure_curvature(img_shape, line, ym_per_pix = 30 / 720, xm_per_pix = 3.7 / 700):
    # Generate y values for plotting
    plot_y = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # Calculate x values using polynomial coeffs
    line_x = line[0] * plot_y ** 2 + line[1] * plot_y + line[2]

    # Evaluate at bottom of image
    y_eval = np.max(plot_y)

    # Fit curves with corrected axis
    curve_fit = np.polyfit(plot_y * ym_per_pix, line_x * xm_per_pix, 2)

    # Calculate curvature for line
    curvature = ((1 + (2 * curve_fit[0] * y_eval * ym_per_pix + curve_fit[1]) ** 2) ** (3 / 2)) / np.absolute(
        2 * curve_fit[0])

    return curvature

def calculate_vehicle_offset(image_shape, left_line, right_line, meter_per_pixel_x  = 3.7 / 700):
    # We will calculate offset at same point as we did for calculating curvature
    y_eval = image_shape[0]

    # Find values for both lines in those at y pos
    left_x = left_line[0] * (y_eval ** 2) + left_line[1] * y_eval + left_line[2]
    right_x = right_line[0] * (y_eval ** 2) + right_line[1] * y_eval + right_line[2]

    # Middle of the lane should be in middle of the image
    mid_image = image_shape[1] // 2

    # Car position - middle between lane lines
    car_position = (left_x + right_x) / 2

    # Calculate offset
    offset = (mid_image - car_position) * meter_per_pixel_x

    return offset

def unwarp_detection(left_line, right_line, inverse_transformation_matrix, undistorted_image):
    # Calculate x points for each y point
    y = np.linspace(0, 720, 719)
    left_line_x_points = left_line[0] * (y ** 2) + left_line[1] * y + left_line[2]
    right_line_x_points = right_line[0] * (y ** 2) + right_line[1] * y + right_line[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undistorted_image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line_x_points, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x_points, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwraped_mask = cv2.warpPerspective(color_warp, inverse_transformation_matrix, (undistorted_image.shape[1], undistorted_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, unwraped_mask, 0.3, 0)
    
    return result

def display_info(image, curvature, offset, straight_line_threshold = 5000):
    average_curve = (curvature[0] + curvature[1]) // 2
    if average_curve >= straight_line_threshold:
        curvature_info = 'Approximately Straight'
    else:
        curvature_info = 'Curvature radius: ' + str(average_curve.astype(np.int)) + 'm'

    cv2.putText(image, (curvature_info), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(image, ('Vehicle offset: ' + str(round(offset, 3)) + 'm'), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image