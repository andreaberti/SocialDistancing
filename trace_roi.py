import pytube

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import os
import re

from matplotlib import patches
from tqdm import tqdm

from utils import *
import argparse
from PIL import Image

# import libraries for videos
import pytube
from IPython.display import HTML


def draw_circle(event, x, y, flags, param):
    '''
    Callback used in cv2 to draw the circles and the lines  of the perimeter
    of the ROI using the mouse click and to take the relevant points useful for the perspective
    transformation.
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    '''
    global edit_frame, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < MAX_MOUSE_POINTS:
            if len(mouse_points) < MAX_POINTS_ROI:
                cv2.circle(edit_frame, (x, y), 5, (0, 0, 255), -1)
            else:
                cv2.circle(edit_frame, (x, y), 5, (255, 0, 0), -1)
            mouse_points.append([x, y])

            # draw line between points
            if len(mouse_points) >= 2 and len(mouse_points) <= MAX_POINTS_ROI:
                cv2.line(edit_frame,
                         (mouse_points[len(mouse_points) - 2][0],
                          mouse_points[len(mouse_points) - 2][1]),
                         (mouse_points[len(mouse_points) - 1][0],
                          mouse_points[len(mouse_points) - 1][1]),
                         (0, 255, 0), 2)
                if len(mouse_points) == MAX_POINTS_ROI:
                    cv2.line(edit_frame, (mouse_points[0][0], mouse_points[0][1]),
                             (mouse_points[len(mouse_points) - 1][0], mouse_points[len(mouse_points) - 1][1]),
                             (0, 255, 0), 2)
                    print(mouse_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to analyze videos and check if people respect social distancing.")
    parser.add_argument("video_file", type=str, help="video file to process. Could be .mp4")
    args = parser.parse_args()

    #get the video name from file
    video_file = args.video_file
    video_name = get_video_name(video_file)
    frames_dir = FRAMES_FOLDER + video_name + "/"
    text_file = VIDEO_FOLDER + video_name + ".txt"
    # check if exist the frames of the video yet
    if not os.path.exists(frames_dir):
        # RIPRENDO IL VIDEO E LO DIVIDO IN FRAME
        video_name, FPS = save_frames_from_video(video_file)
    else:
        video_name, FPS, width, height, _, _ = read_results(text_file)

    #collect frames in list
    frames = get_frames(frames_dir)

    # first_frame = cv2.imread(FRAMES_FOLDER + video_name + "/" + file_frame)
    mouse_points = []
    index_frame = 0
    #read first frame
    frame = cv2.imread(frames_dir + frames[index_frame])
    height, width, _ = frame.shape
    print("width e height ", width, height)

    # set window and callback to trace ROI
    cv2.namedWindow("Trace points of ROI")
    cv2.setMouseCallback("Trace points of ROI", draw_circle)
    edit_frame = frame.copy()
    # visualize window and start to trace ROI
    # trace points must follow bottom-left, bottom-right, top-right, top-left order
    while True:
        cv2.imshow("Trace points of ROI", edit_frame)
        key_pressed = cv2.waitKey(20)
        # press ESC to stop drawing
        if key_pressed == 27:
            exit(0)
        # clean image and restart to take mouse points
        # if 'r' button is pressed
        if key_pressed == 114:
            mouse_points = []
            edit_frame = frame.copy()

        #space button to save results
        if key_pressed == 32:
            edit_frame = frame.copy()
            break

        # n 'next' button to skip to next frame
        if key_pressed == 110:
            index_frame = (index_frame + 1) % (len(frames))
            frame = cv2.imread(frames_dir + frames[index_frame])
            edit_frame = frame.copy()
            mouse_points = []

        # b 'back' button to skip previous frame
        if key_pressed == 98:
            index_frame = len(frames) - 1 if index_frame == 0 else index_frame - 1
            frame = cv2.imread(frames_dir + frames[index_frame])
            edit_frame = frame.copy()
            mouse_points = []


    cv2.destroyWindow("Trace points of ROI")

    # end to trace ROI, write the results on a text file
    write_results(text_file, video_name, FPS, width, height, mouse_points)

    video_name, FPS, width, height, points_ROI, points_distance = read_results(text_file)

    print("video name :", video_name)
    print("FPS :", FPS)
    print("width :", width)
    print("height :", height)
    print("points of ROI :", points_ROI)
    print("points of distance :", points_distance)
