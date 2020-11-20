import numpy as np
import cv2
import os
import re
import pytube
import matplotlib.pyplot as plt

# DEFINE SOME CONSTANTS
VIDEO_FOLDER = "./video/"
FRAMES_FOLDER = "./frames/"
MAX_MOUSE_POINTS = 7
MAX_POINTS_ROI = 4
MAX_DANGEROUS_DISTANCE = 150
MAX_WARNING_DISTANCE = 180
COLOR_SAFE = (0,255,0)
COLOR_WARNING = (0,255,255)
COLOR_DANGEROUS = (0,0,255)


def download_from_youtube(video_url, folder_path, video_name=None):
    '''
    Method used to download video from Youtube
    :param video_url: the url of the video to download
    :param folder_path: the folder path where save the video
    :param video_name: the name to give to the save file
    :return:
    '''
    youtube = pytube.YouTube(video_url)
    video = youtube.streams.first()

    video.download(folder_path)  # path, where to video download.
    # #if u want to rename the video
    # if not video_name is None:
    #    os.rename(folder_path + video.title + ".mp4", video_name + ".mp4")
    #    return video_name
    return video.title


def get_frame_rate(video_capture):
    '''
    Method that return the FPS of a video
    :param video_capture: the Videocapture object of a video
    :return: the number of FPS
    '''
    FPS = video_capture.get(cv2.CAP_PROP_FPS)
    print("frame_rate: " + str(FPS))
    return FPS


def save_frames_from_video(video_path, max_frames = 750):
    '''
    Method that takes the path of a video, read the video
    and create all the frames of the video, saving them on
    the specific folder of the video inside the frames folder
    :param video_path: path of the input video
    :return: the video name and FPS
    '''
    if not os.path.isfile(video_path):
        IOError("File video doesn't exists!")
        return

    # check if exists frames dir, otherwise create it
    if not os.path.isdir(FRAMES_FOLDER):
        os.mkdir(FRAMES_FOLDER)

    # take video name to rename save folder
    video_name = get_video_name(video_path)

    # define where save the frames and create
    # the folder if not exists yet
    save_path_folder = FRAMES_FOLDER + video_name + "/"
    if not os.path.isdir(save_path_folder):
        os.mkdir(save_path_folder)

    print("Save frames from " + video_path + " ...")
    # capture video
    cap = cv2.VideoCapture(video_path)
    cnt = 0

    # check frame rate
    FPS = get_frame_rate(cap)

    # Check if video file is opened successfully
    if (cap.isOpened() == False):
        IOError("Error opening video stream or file")

    # read first frame
    ret, first_frame = cap.read()

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            # save each frame to folder
            cv2.imwrite(save_path_folder + str(cnt) + '.png', frame)
            cnt = cnt + 1

            if cnt >= max_frames:
                print("Done! " + str(cnt) + " frames saved in" + save_path_folder)
                return video_name, FPS

        # Break the loop
        else:
            print("Done! " + str(cnt) + " frames saved in" + save_path_folder)
            return video_name, FPS


def get_frames(frames_dir):
    '''
    Method that returns from frames_folder passed in input
    the sorted list of frame_names
    :param frames_dir: the path of the frames_dir
    :return: the sorted list of frame_names
    '''
    frames = os.listdir(frames_dir)
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    return frames


def get_video_name(video_path):
    '''
    Method that returns the video name from path
    :param video_path: path of the video
    :return: the video name
    '''
    return video_path.split("/")[-1].split(".")[0]


def compute_perspective_transform(corner_points, width, height, image):
    ''' Compute the transformation matrix useful for the bird eye view image
	:param corner_points : 4 corner points selected from the image, that indicates the ROI
	:param  height, width : size of the image
	return : transformation matrix and the transformed image
	'''
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    # order is left-bottom, right-bottom, right-top, left-top
    img_params = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_perspective_unit_distances(unit_points, matrix_transformation):
    '''
    Compute the perspective trasformation of the points used to take the right distances.
    Points must be: central, width, height order
    :param unit_points: list of 3 points
    :param matrix_transformation: matrix trasformation used to transform the points
    :return: the distances in horizontal and vertical used to calculates distance between two humans
    through the midpoints
    '''
    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    points_distance = np.float32(np.array([unit_points]))
    warped_pt = cv2.perspectiveTransform(points_distance, matrix_transformation)[0]

    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    return distance_w, distance_h


def compute_point_perspective_transformation(matrix, list_midpoints):
    ''' Apply the perspective transformation to every ground point which have been detected on the main frame.
	:param  matrix : the 3x3 matrix
	:param  list_midpoints : list that contains the points to transform
	return : list containing all the new points
	'''
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_midpoints).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list


def mid_point(person):
    '''
    Method used to return the bottom midpoint of a bbox of a singular person
    :param person: the bbox of the person
    :return: the bottom midpoint (x,y)
    '''
    # get the coordinates
    x1, y1, x2, y2 = person
    # compute bottom center of bbox
    x_mid = int((x1 + x2) / 2)
    y_mid = int(y2)
    mid = (x_mid, y_mid)
    return mid


def calculate_euclidean_distance(p1, p2, distance_w=None, distance_h=None):
    '''
    Method used to calculate euclidean distance between two people represented by bottom midpoints.
    THe method can calculate both the normal euclidean distance if are not turned in input
    the horizontal distance and the vertical distance calculated using the perspective transformation,
    and also the euclidean distance of the midpoints after the transformation through the matrix perspective
    transformation.
    :param p1: first people represented by a midpoint (x1,y1)
    :param p2: second people represented by a midpoint (x1,y1)
    :param distance_w: if is None, calculate euclidean distance over midpoints taken from the frame, otherwise
    calculate distance after perspective transformation of the midpoints on bird eye view
    :param distance_h: if is None, calculate euclidean distance over midpoints taken from the frame, otherwise
    calculate distance after perspective transformation of the midpoints on bird eye view
    :return: the euclidean distance between p1 and p2
    '''
    if distance_h is not None and distance_w is not None:
        h = abs(p2[1] - p1[1])
        w = abs(p2[0] - p1[0])

        dis_w = float((w / distance_w) * 180)
        dis_h = float((h / distance_h) * 180)

        return int(np.sqrt(((dis_h) ** 2) + ((dis_w) ** 2)))
    else:
        return int(np.sqrt(((p2[1] - p1[1]) ** 2) + ((p2[0] - p1[0]) ** 2)))


def compute_distances(midpoints, distance_w=None, distance_h=None):
    '''
    Method that takes in input a list of all people of a frame (represented by midpoints)
    and calculate for each pair of midpoints, the euclidean distance between us. The method
    creates a list of distance between people where each people is indicated by tuple (index_of_midpoint, distance).
    The method sort the tuples by distances. Is also calculated a list of distance line composed by tuple of
    (index_first_person,index_second_person,distance) and then is sorted also this list.
    :param midpoints: list of midpoints (x1,y1)
    :param distance_w: if is None, calculate euclidean distance over midpoints taken from the frame, otherwise
    calculate distance after perspective transformation of the midpoints on bird eye view
    :param distance_h: if is None, calculate euclidean distance over midpoints taken from the frame, otherwise
    calculate distance after perspective transformation of the midpoints on bird eye view
    :return: a sorted list of distances and a sorted list of distances line.
    '''
    num_people = len(midpoints)
    distances = []
    distancesLine = []
    for i in range(num_people):
        for j in range(i + 1, num_people):
            if i < j:
                dist = calculate_euclidean_distance(midpoints[i], midpoints[j],
                                                    distance_w, distance_h)

                # add to list
                distances.append((i, dist))
                distances.append((j, dist))
                distancesLine.append((i, j, dist))

    sorted_distances = sorted(distances, key=lambda tup: tup[1])
    sorted_dist_line = sorted(distancesLine, key=lambda tup: tup[2])

    return sorted_distances, sorted_dist_line


def return_people_ids(bboxes):
    '''
    Method that takes the bboxes of the people detected
    to check how many people are present in a frame
    and create a list of indices of people.
    :param bboxes: a list of tuple (x1,y1,x2,y2)
    :return: a list of people indices
    '''
    people_ids = [x for x in range(len(bboxes))]
    return people_ids


def check_risks_people(distances, people_ids):
    '''
    Method used to separate people detected in the right sets based on the order of the distances in the list
    turned in input.
    There are three sets: safe,warning,dangerous.
    Each person can be only in one set. If the person has a distance less then the MAX_DANGEROUS_DISTANCE, is placed
    inside the dangerous set, else if the distance is between the MAX_DANGEROUS_DISTANCE and the MAX_WARNING_DISTANCE, is
    paced inside the warning set, otherwise in safe set.

    :param distances: a sorted list composed by (index_of_person,distance_between_other_person)
    :param people_ids: a list of the indices of the person
    :return: 3 sets composed by the indices of the people
    '''
    set_safe = set()
    set_warning = set()
    set_dangerous = set()

    list_people = people_ids

    # if is detected only one person, put directly in safe set
    if len(list_people) == 1:
        set_safe.add(0)
        return set_safe, set_warning, set_dangerous

    # otherwise assign each person based on distance in the right set
    for d in distances:
        p, dist = d
        if len(list_people) == 0:
            break

        if p in list_people:
            if dist <= MAX_DANGEROUS_DISTANCE:
                set_dangerous.add(p)
            elif dist > MAX_DANGEROUS_DISTANCE and dist <= MAX_WARNING_DISTANCE:
                set_warning.add(p)
            else:
                set_safe.add(p)

            list_people.remove(p)

    return set_safe, set_warning, set_dangerous


def write_results(file_txt, video_name, FPS, width, height, mouse_points):
    '''
    Method used to save all the information after the trace_ROI operation on the frame.
    :param file_txt: file where write the results
    :param video_name: the name of the video
    :param FPS: the number of Frame per seconds
    :param width: the width of the frame
    :param height: the heigth of the frame
    :param mouse_points: a list of points where the first four points indicate the ROI traced on the frame
    and the last three points indicates the unit distances.
    '''
    # create file .txt and write results
    f = open(file_txt, "w+")

    f.write("video name:" + video_name + '\n')
    f.write("FPS:" + str(FPS) + '\n')
    f.write("width:" + str(width) + '\n')
    f.write("height:" + str(height) + '\n')
    f.write("points of ROI:\n")
    for p in mouse_points[:4]:
        x, y = p
        f.write(str(x) + "," + str(y) + "\n")

    f.write("points of distance:\n")
    for p in mouse_points[4:7]:
        x, y = p
        f.write(str(x) + "," + str(y) + "\n")
    f.close()


def read_results(file_txt):
    '''
    Method used to read the info of specific video, like the name, the FPS, width,height,
    points of ROI, points of unit distance.
    :param file_txt: file where read the informations
    :return: name of the video, the FPS, width, height, points_of ROI, points of unit distances.
    '''
    b_take_ROI = False
    b_take_distance = False
    points_ROI = []
    points_distance = []
    video_name = ""
    FPS = 0.0
    width = 0
    height = 0

    with open(file_txt) as fr:
        for line in fr.readlines():
            if "video_name" in line:
                video_name = line.split(":")[1]
            elif "FPS:" in line:
                FPS = float(line.split(":")[1])
            elif "width:" in line:
                width = int(line.split(":")[1])
            elif "height:" in line:
                height = int(line.split(":")[1])
            elif "points of ROI:" in line:
                b_take_ROI = True
            elif "points of distance:" in line:
                b_take_ROI = False
                b_take_distance = True
            elif b_take_ROI:
                point = line.split(",")
                points_ROI.append([int(point[0]), int(point[1])])
            elif b_take_distance:
                point = line.split(",")
                points_distance.append([int(point[0]), int(point[1])])

        fr.close()
    return video_name, FPS, width, height, points_ROI, points_distance


def create_video(frames_dir, video_name, FPS):
    '''
    Method that read all the edit frames inside the directory, sort them
    and then create a video.
    :param frames_dir: path of the folder where get the frames
    :param video_name: the name of the video to create
    :param FPS: the number of FPS used to compose the video
    '''
    frames = os.listdir(frames_dir)
    frames.sort(key=lambda f: int(re.sub('\D', '', f)))
    frame_array = []

    for i in range(len(frames)):
        # reading each files
        img = cv2.imread(frames_dir + frames[i])
        height, width, _ = img.shape
        size = (width, height)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def resize_and_pad(img, size, pad_color=0):
    '''
    Method used to resize the bird eye view to fit on the final mosaic output.
    :param img: the bird eye image
    :param size: size to set the image resized
    :param pad_color: color used to fill the pad around the image
    :return: the bird eye view image resized
    '''
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color,(list, tuple, np.ndarray)):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


def create_plot_contagion(contagion_map, title, modality="h"):
    '''
    Method used to create a plot of contagion during the video analysis and save it on image.
    The plot is organized in --> x: frames, y: n_of people.
    The plot draw the lines based on the number of people detected in each frame, the number of people in the various
    state divided by colors.
    :param contagion_map: a list that contains a number of tuples composed by (n_people_detected,n_safe,n_low_risk,n_high_risk)
    equal to the number of frames.
    :param title: title to give to the plot and used to save the image.
    :param modality: is possible to compose a string of modality.
    The options are:
        - 'h' --> draw the line of high risk people (red line)
        - 'l' --> draw the line of low risk people (yellow line)
        - 's' --> draw the line of safe people (green line)
    Is possible to create different compositions with options.
    Example --> 'hl' draw lines relative to high risk people and low risk people ( red, yellow lines).
    '''
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=[27, 9])
    ax = plt.axes()
    # set info axes
    plt.title(title, fontsize=20)
    plt.xlabel("frames", fontsize=18)
    plt.ylabel("n° people", fontsize=18)

    people_detected = []
    safe_people = []
    warning_people = []
    dangerous_people = []

    # x are frames
    frames = np.arange(0, len(contagion_map))

    # get data from contagion_map
    for f in frames:
        p, s, w, d = contagion_map[f]
        # set in right lists
        people_detected.append(p)
        safe_people.append(s)
        warning_people.append(w)
        dangerous_people.append(d)

    # set limits to plot
    plt.xlim(0, len(contagion_map))
    plt.ylim(min(people_detected), max(people_detected) + 2)

    # plot based on modality chosen
    plt.plot(frames, people_detected, color='black', linestyle='-', label="detected")
    if 'h' in modality:
        plt.plot(frames, dangerous_people, color="red", linestyle='-', label="high-risk")
    if 'l' in modality:
        plt.plot(frames, warning_people, color="yellow", linestyle='-', label="low-risk")
    if 's' in modality:
        plt.plot(frames, safe_people, color="green", linestyle='-', label="safe")

    # set legend and save figure plot
    plt.legend(fontsize=18)
    fig.savefig(title + '.jpg')



