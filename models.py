import os
from copy import deepcopy

import detectron2
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

# from utils import mid_point, get_frames, FRAMES_FOLDER, compute_perspective_transform, \
#   compute_perspective_unit_distances, return_people_ids, compute_point_perspective_transformation, compute_distances, \
#   check_risks_people, COLOR_SAFE, COLOR_WARNING, COLOR_DANGEROUS

# DA PROVARE SU COLAB
from SocialDistancing import utils

# import utils


setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import libraries for Yolo
from imageai.Detection import ObjectDetection


def faster_RCNN_model():
    '''
    Method that creates a fasterRCNN model, using config and pretrained weights
    from detectron2 core library
    :return: the fasterRCNN model used to predict
    '''
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.94  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def YoloV3_model(yolov3_model_path, b_tiny_version=False):
    '''
    Method that creates a YoloV3 model, using config from ImageAi core library.
    :param yolov3_model_path: the path of the config file
    :return: the YoloV3 model used to predict and the custom objects ( only poeple) to pass to the model
    during prediction.
    '''
    detector = ObjectDetection()
    if not b_tiny_version:
        detector.setModelTypeAsYOLOv3()
    else:
        detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(yolov3_model_path)
    custom_objects = detector.CustomObjects(person=True)
    detector.loadModel()
    return detector, custom_objects


def find_people_fasterRCNN(frame, model):
    '''
    Method used to do detection of people with fasterRCNN model.
    :param frame: input frame where perform prediction.
    :param model: the model used to predict on frame.
    :return: the people bounding boxes [x1,y1,x2,y2] and the bottom midpoints [x1,y1]
    '''
    # img = cv2.imread(frame_file)
    outputs = model(frame)
    classes = outputs['instances'].pred_classes.cpu().numpy()
    bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    ind = np.where(classes == 0)[0]
    people = bbox[ind]
    midpoints = [utils.mid_point(person) for person in people]
    return people, midpoints


def find_people_YoloV3(frame, model, custom_objects=None):
    '''
    Method used to do detection of people with YoloV3 model.
    :param frame: input frame where perform prediction.
    :param model: the model used to predict on frame.
    :param custom_objects: indicates to the model to detect only people
    :return: the people bounding boxes [x1,y1,x2,y2] and the bottom midpoints [x1,y1]
    '''
    returned_image, detections = model.detectCustomObjectsFromImage(
        custom_objects=custom_objects,
        input_type="array", input_image=frame,
        output_type="array",
        minimum_percentage_probability=30
    )
    people = [x['box_points'] for x in detections]
    midpoints = [utils.mid_point(person) for person in people]
    return people, midpoints


def perform_social_detection(video_name, points_ROI, points_distance, width, height, selected_model):
    '''
    Papeline of social detection prediction that read all frames of the video passed in input
    finds the people inside in each frame based on the selected prediction model,
    and compute the perspective trasformation of the midpoints on the bird eye view.
    Then for each midpoint transformed, compute the euclidean distance to check which
    are the distances between the relative midpoints.
    Finally color the bboxes of the people and the points on bird eye view image of the
    right social distancing color and save the edit frame. During the social detection is created also
    a contagion map that indicates for each frames how many contagions there are.
    :param video_name: the name of the video
    :param points_ROI: list of 4 points of ROI
    :param points_distance: list of 3 points relative to calibrate the distances inside the bird eye view
    :param width: the width of the frames
    :param height: the height of the frames
    :param selected_model: the model used to perform the people detection. Can be chosen 'yolo' , 'fasterRCNN' or 'yolo-tiny' options.
    :return: the contagion map composed by tuples (n people detected,n safe, n warning,n dangeorous)
    for how many frames there are in the video
    '''

    print("Processing ", video_name, "...")
    output_folder = "out/"
    contagion_map = []

    # create output folder of processed frames
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_folder + video_name):
        os.mkdir(output_folder + video_name)

    # get frames to be processed
    frames = utils.get_frames(utils.FRAMES_FOLDER + video_name)

    # first frame
    frame_file = utils.FRAMES_FOLDER + video_name + "/" + frames[0]
    img = cv2.imread(frame_file)

    # get perspective transformation points of ROI and distance
    matrix_transformation, bird_eye_frame = utils.compute_perspective_transform(points_ROI, width, height, img)
    distance_w, distance_h = utils.compute_perspective_unit_distances(points_distance, matrix_transformation)

    # define model based on parameter 'selected model'
    if selected_model == 'yolo':
        print("find distances with YoloV3")
        yolov3_model_path = "./yolo.h5"
        model, custom_objects = YoloV3_model(yolov3_model_path)
    elif selected_model == 'yolo-tiny':
        print("find distances with YoloV3-tiny")
        yolov3_tiny_model_path = "./yolo-tiny.h5"
        model, custom_objects = YoloV3_model(yolov3_tiny_model_path, b_tiny_version=True)
    elif selected_model == 'fasterRCNN':
        print("find distances with fasterRCNN")
        model = faster_RCNN_model()

    # get info from bird eye frame
    bird_height, bird_width, _ = bird_eye_frame.shape
    # size for resize the bird eye
    dsize = (width - 100, height)

    # process over the frames
    for f in tqdm(frames):
        # read the frame and create a copy of background image
        frame = cv2.imread(utils.FRAMES_FOLDER + video_name + "/" + f)

        # create bird-eye-view image
        bird_eye_view_img = np.zeros((bird_height, bird_width, 3))

        # choose the right predict based on 'selected_model' parameter
        if selected_model == 'yolo' or selected_model == 'yolo-tiny':
            bboxes, midpoints = find_people_YoloV3(frame, model, custom_objects)
        elif selected_model == 'fasterRCNN':
            bboxes, midpoints = find_people_fasterRCNN(frame, model)

        # return the indices of the people detected
        people_ids = utils.return_people_ids(bboxes)

        # perform operations on frame if is detected at least 1 person
        if len(midpoints) > 0:
            # transform midpoints based on the matrix perspective transformation
            # calculate the distances on bird eye
            midpoints_transformed = utils.compute_point_perspective_transformation(matrix_transformation, midpoints)
            dist_bird, dist_line = utils.compute_distances(midpoints_transformed, distance_w, distance_h)

            # divide the people in the right sets based on the distance calculated
            set_safe_faster, set_warning_faster, set_dangerous_faster = utils.check_risks_people(dist_bird, people_ids)

            # Draw the boxes on the frame based on the warning degree
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                if i in set_safe_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), utils.COLOR_SAFE)
                elif i in set_warning_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), utils.COLOR_WARNING)
                elif i in set_dangerous_faster:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), utils.COLOR_DANGEROUS)

            # Draw circles of right color on bird eye image
            for i in range(len(midpoints_transformed)):
                x, y = midpoints_transformed[i][0], midpoints_transformed[i][1]
                if i in set_safe_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 3, utils.COLOR_SAFE, 3)
                elif i in set_warning_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 3, utils.COLOR_WARNING, 3)
                elif i in set_dangerous_faster:
                    cv2.circle(bird_eye_view_img, (x, y), 3, utils.COLOR_DANGEROUS, 3)

            # draw distance lines on frame
            # draw distance lines on bird eye view
            for line in dist_line:
                p1, p2, d = line

                if d <= utils.MAX_DANGEROUS_DISTANCE:
                    cv2.line(frame, tuple(midpoints[p1]), tuple(midpoints[p2]), utils.COLOR_DANGEROUS)
                    cv2.line(bird_eye_view_img, tuple(midpoints_transformed[p1]),
                             tuple(midpoints_transformed[p2]), utils.COLOR_DANGEROUS)
                elif utils.MAX_DANGEROUS_DISTANCE < d <= utils.MAX_WARNING_DISTANCE:
                    cv2.line(frame, tuple(midpoints[p1]), tuple(midpoints[p2]), utils.COLOR_WARNING)
                    cv2.line(bird_eye_view_img, tuple(midpoints_transformed[p1]),
                             tuple(midpoints_transformed[p2]), utils.COLOR_WARNING)

            # set text to write on background image based on statistics
            text_number_people = "People detected: " + str(len(midpoints_transformed))
            text_safe = "People safe: " + str((len(set_safe_faster) / len(midpoints_transformed)) * 100) + "%"
            text_warning = "People low risk: " + str(
                (len(set_warning_faster) / len(midpoints_transformed)) * 100) + "%"
            text_dangerous = "People high risk: " + str(
                (len(set_dangerous_faster) / len(midpoints_transformed)) * 100) + "%"

            # fill contagion_map --> (n_people, n_safe, n_warning, n_dangerous)
            contagion_tuple = (len(midpoints), len(set_safe_faster), len(set_warning_faster), len(set_dangerous_faster))
            contagion_map.append(contagion_tuple)
        else:
            # no people detected, write only 0 on the background image
            text_number_people = "People detected: 0"
            text_safe = "People safe: 0.0%"
            text_warning = "People low risk: 0.0%"
            text_dangerous = "People high risk: 0.0%"

            # fill contagion_map with zeros
            contagion_tuple = (0, 0, 0, 0)
            contagion_map.append(contagion_tuple)

        # scale bird-eye-img
        bird_eye_view_img = utils.resize_and_pad(bird_eye_view_img, dsize)

        # create background image for text
        background_height = bird_eye_view_img.shape[0] - height
        background_width = width

        background_img = np.zeros((background_height, background_width, 3), dtype=np.uint8)
        background_img[:background_height, :background_width] = (127, 127, 127)

        # set text on image
        cv2.putText(background_img, text_number_people, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                    2, cv2.LINE_4)
        cv2.putText(background_img, text_safe, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, utils.COLOR_SAFE, 2,
                    cv2.LINE_4)
        cv2.putText(background_img, text_warning, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, utils.COLOR_WARNING,
                    2, cv2.LINE_4)
        cv2.putText(background_img, text_dangerous, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    utils.COLOR_DANGEROUS, 2, cv2.LINE_4)

        # compose the image
        numpy_vertical = np.vstack((frame, background_img))
        numpy_vertical_concat = np.concatenate((frame, background_img), axis=0)
        numpy_horizontal = np.hstack((numpy_vertical_concat, bird_eye_view_img))
        numpy_horizontal_concat = np.concatenate((numpy_vertical_concat, bird_eye_view_img), axis=1)

        # write result of edit frame
        cv2.imwrite(output_folder + video_name + "/" + f, numpy_horizontal_concat)

    # return contagion map
    return contagion_map
