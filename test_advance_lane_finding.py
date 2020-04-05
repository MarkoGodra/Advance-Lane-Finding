import lane_lines
from moviepy.editor import VideoFileClip

def process_image(image):
    result = lane_lines_finder.run(image)
    return result

lane_lines_finder = lane_lines.LaneFinder('calibration_output/wide_dist_pickle.p')

video = 'project_video'
input_video = video + '.mp4'
output_video = 'out_videos/' + video + '_out.mp4'
white_output = output_video
clip1 = VideoFileClip(input_video)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
