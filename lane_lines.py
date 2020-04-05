import numpy as np
import pickle
import advance_lane_finding as alf

class LaneLine:
    def __init__(self):
        # Indicator wheater line is detected
        self.detected = False

        # Parameters of curve
        self.coeffs = None

        # Use averaging over 3 latest fits
        self.previous_fits = []

        self.previous_fits_size = 3

        # Index for latest fit
        self.previous_fits_index = 0 % self.previous_fits_size

        # Curvature
        self.curvature = 0

        # Initialized
        self.ring_buffer_filled = False

    def is_detected(self):
        return self.detected

    def get_averaged_line(self):
        if self.ring_buffer_filled is True:
            return (np.sum(self.previous_fits, axis = 0) / self.previous_fits_size)
        else:
            # If we do not have history just return current calculation
            return self.coeffs

    def update_previous_values(self):
        if len(self.previous_fits) < self.previous_fits_size:
            self.previous_fits.append(self.coeffs)
        else:
            # Our buffer is now full
            self.ring_buffer_filled = True

            # Update list of previous indexes
            self.previous_fits[self.previous_fits_index] = self.coeffs

            # Update write index for previous fits
            self.previous_fits_index  = (self.previous_fits_index + 1) % self.previous_fits_size

    def update(self, line_x, line_y):
        # Check first
        if (line_x.size > 0) & (line_y.size > 0):
            # Left line is detected
            self.detected = True
            self.coeffs = alf.fit_poly(line_x, line_y)
            self.update_previous_values()
        else:
            # If we fail to detected line, do not update, use previous value, but mark lane as not detected for next iteration
            self.detected = False

class LaneFinder:
    def __init__(self, calibration_params_path):
        super().__init__()

        # Load coeffs
        dist_pickle = pickle.load(open(calibration_params_path, "rb" ))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

        self.left_line = LaneLine()
        self.right_line = LaneLine()

    def run(self, image):
        # Undistort image
        undistorted_image = alf.undistort_image(image, self.mtx, self.dist)

        # Convert image to binary
        binary_image = alf.apply_thresholds(image, 
            abs_soble_threshold = (20, 100),
            color_thresholds=(130, 255)
            )

        # Change perspective
        warped_image, invers_transform_mat = alf.go_to_birdview_perspective(binary_image)

        if self.left_line.is_detected() & self.right_line.is_detected():
            # If we have previous detection, do targeted search
            left_line_x, left_line_y, right_line_x, right_line_y = alf.targeted_search(warped_image, self.left_line.coeffs, self.right_line.coeffs)
            self.left_line.update(left_line_x, left_line_y)
            self.right_line.update(right_line_x, right_line_y)
        else:
            # Do blind search
            left_line_x, left_line_y, right_line_x, right_line_y = alf.find_lane_pixels(warped_image)
            self.left_line.update(left_line_x, left_line_y)
            self.right_line.update(right_line_x, right_line_y)

        # Get averaged lane lines
        left_line_coeffs = self.left_line.get_averaged_line()
        right_line_coeffs = self.right_line.get_averaged_line()

        # Calculate curvature radius
        self.left_line.curvature = alf.measure_curvature(warped_image.shape, left_line_coeffs)
        self.right_line.curvature = alf.measure_curvature(warped_image.shape, right_line_coeffs)

        # Calculate vehicle offset
        vehicle_offset = alf.calculate_vehicle_offset(warped_image.shape, left_line_coeffs, right_line_coeffs)

        # Draw lines onto undistorted image
        output = alf.unwarp_detection(left_line_coeffs, right_line_coeffs, invers_transform_mat, undistorted_image)

        # Display text info on image
        alf.display_info(output, (self.left_line.curvature, self.right_line.curvature), vehicle_offset)

        return output
