import advance_lane_finding as alf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import glob

# Calibrate camera
alf.calibrate_camera()

# Load coeffs
dist_pickle = pickle.load(open('calibration_output/wide_dist_pickle.p', "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

images_path = 'test_images/test3.jpg'
image = cv2.imread(images_path)

# Undistort image
undistorted = alf.undistort_image(image, mtx, dist)
cv2.imwrite('output_images/undistorted0.jpg', undistorted)

# Convert image to binary
binary_image = alf.apply_thresholds(undistorted, 
    abs_soble_threshold = (20, 100),
    color_thresholds=(130, 255)
    )
cv2.imwrite('output_images/binary_thresholding_full.jpg', np.dstack([binary_image, binary_image, binary_image]) * 255)

# Seprate thresholding results
abs_sobel = alf.absolute_sobel_thresholding(undistorted)
cv2.imwrite('output_images/abs_sobel.jpg', np.dstack([abs_sobel, abs_sobel, abs_sobel]) * 255)

color_binary = alf.color_thresholding(undistorted)
cv2.imwrite('output_images/color_binary.jpg', np.dstack([color_binary, color_binary, color_binary]) * 255)

gradient_mag = alf.gradient_magnitude_thresholding(undistorted, mag_thresh=(80, 100))
cv2.imwrite('output_images/gradient_mag.jpg', np.dstack([gradient_mag, gradient_mag, gradient_mag]) * 255)

gradient_dir = alf.gradient_direction_thresholding(undistorted, thresh=(0.7, 1.4))
cv2.imwrite('output_images/gradient_dir.jpg', np.dstack([gradient_dir, gradient_dir, gradient_dir]) * 255)

# Change perspective
warped_image, invers_transform_mat = alf.go_to_birdview_perspective(binary_image)
cv2.imwrite('output_images/warped_image.jpg', np.dstack([warped_image, warped_image, warped_image]) * 255)

left_line_x, left_line_y, right_line_x, right_line_y, window_output = alf.find_lane_pixels(warped_image, draw_windows=True)
cv2.imwrite('output_images/window_output.jpg', window_output)

# This function writes internaly
left_line, right_line, fitted_lines_window = alf.fit_polynomial(left_line_x, left_line_y, right_line_x, right_line_y, window_output.shape, window_output, draw_lines=True)

# Targeted search
left_line_x, left_line_y, right_line_x, right_line_y, targeted_out = alf.targeted_search(warped_image, left_line, right_line, draw_lines=True)

# Calculate curvature radius
left_line_curvature = alf.measure_curvature(warped_image.shape, left_line)
right_line_curvature = alf.measure_curvature(warped_image.shape, right_line)

# # Calculate vehicle offset
vehicle_offset = alf.calculate_vehicle_offset(warped_image.shape, left_line, right_line)

# # Draw lines onto undistorted image
output = alf.unwarp_detection(left_line, right_line, invers_transform_mat, undistorted)
cv2.imwrite('output_images/final_output.jpg', output)

# Display text info on image
alf.display_info(output, (left_line_curvature, right_line_curvature), vehicle_offset)
cv2.imwrite('output_images/final_output_with_info.jpg', output)
