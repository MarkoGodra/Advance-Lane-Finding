import advance_lane_finding as alf
import pickle
from moviepy.editor import VideoFileClip

def process_image(image):
    result = pipeline(image)
    return result

def pipeline(image):

    # Undistort image
    undistorted_image = alf.undistort_image(image, mtx, dist)

    # Convert image to binary
    binary_image = alf.apply_thresholds(undistorted_image)

    # Change perspective
    wraped_image, invers_transform_mat = alf.go_to_birdview_perspective(binary_image)

    # Find lanes and fit poly lines
    left_line, right_line = alf.fit_polynomial(wraped_image)

    # Calculate lane curvature
    left_curvature = alf.measure_curvature(left_line, wraped_image.shape)
    right_curvature = alf.measure_curvature(right_line, wraped_image.shape)

    # Calculate vehicle offset
    vehicle_offset = alf.calculate_vehicle_offset(wraped_image.shape, left_line, right_line)

    # Draw lines onto undistorted image
    output = alf.unwarp_detection(left_line, right_line, invers_transform_mat, undistorted_image)

    # Display text info on image
    alf.display_info(output, (left_curvature, right_curvature), vehicle_offset)

    print(left_curvature)
    print(right_curvature)
    print(vehicle_offset)

    return output



dist_pickle = pickle.load(open('calibration_output/wide_dist_pickle.p', "rb" ))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

white_output = 'out_videos/project_video_short_out.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4").subclip(0, 10)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)