import argparse
import pickle
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from search import *
from heatmap import *

dist_pickle = pickle.load(open("svc_pickle.p", "rb" ))

color_space = dist_pickle["color_space"]
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hog_channel = dist_pickle["hog_channel"]
hist_bins = dist_pickle["hist_bins"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]


def process_image(image):
    window_image = np.copy(image)
    image = image.astype(np.float32)/255

    x_start_stop = [None, None]
    xy_overlap = (0.75, 0.75)

    y_start_stops = [[400, 645], [400, 600], [400, 550]]
    xy_windows = [(128, 128), (96, 96), (64, 64)]

    windows = []

    for y_start_stop, xy_window in zip(y_start_stops, xy_windows):
        windows.extend(slide_window(image, x_start_stop, y_start_stop, xy_window, xy_overlap))

    hot_windows = search_windows(image, windows, svc, X_scaler,
                        color_space='YCrCb', spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(window_image), labels)

    return draw_img

def process_video(video_filename, prefix="processed_"):
    clip = VideoFileClip(video_filename)
    new_clip = clip.fl_image(process_image)
    new_filename = prefix + video_filename
    new_clip.write_videofile(new_filename, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create processed video.')
    parser.add_argument(
        'video_filename',
        type=str,
        default='',
        help='Path to video file.'
    )
    args = parser.parse_args()

    video_filename = args.video_filename
    process_video(video_filename)
