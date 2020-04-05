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
        self.previous_fits = [None, None, None]

        # Index for latest fit
        self.previous_fits_index = 0 % 3

        # Curvature
        self.curvature = 0

    def is_detected(self):
        return self.detected

    def get_averaged_line(self):
        # TODO MAKE AVERAGE LOGIC
        return np.sum(self.previous_fits) / 3

    def update(self, line_x, line_y):
        # Check first
        if (line_x.size > 0) & (line_y.size > 0):
            # Left line is detected
            self.detected = True
            self.coeffs = alf.fit_poly(line_x, line_y)
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

        # TODO: TUNE PARAMS HERE
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

        # TODO IMPLEMENT SANITY CHECK

        # Calculate lane curvature
        self.left_line.curvature = alf.measure_curvature(self.left_line.coeffs, warped_image.shape)
        self.right_line.curvature = alf.measure_curvature(self.right_line.coeffs, warped_image.shape)

        # Calculate vehicle offset
        vehicle_offset = alf.calculate_vehicle_offset(warped_image.shape, self.left_line.coeffs, self.right_line.coeffs)

        # Draw lines onto undistorted image
        output = alf.unwarp_detection(self.left_line.coeffs, self.right_line.coeffs, invers_transform_mat, undistorted_image)

        # Display text info on image
        alf.display_info(output, (self.left_line.curvature, self.right_line.curvature), vehicle_offset)

        return output

