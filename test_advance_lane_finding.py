import lane_lines
from moviepy.editor import VideoFileClip

def process_image(image):
    result = lane_lines_finder.run(image)
    # result = image
    return result

lane_lines_finder = lane_lines.LaneFinder('calibration_output/wide_dist_pickle.p')

white_output = 'out_videos/challenge_video_out.mp4'
# white_output = 'out_videos/project_challenge_out2.mp4'
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("project_video.mp4").subclip(36, 45)
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)