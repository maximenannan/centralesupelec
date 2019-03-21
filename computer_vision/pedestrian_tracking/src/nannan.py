import os
from typing import List
from typing import Tuple

import cv2
import numpy as np

from model import Model


def read_frames(images_dir: str) -> List:
    """
    Read frames
    This function sorts frames by their name which can create issue if names are not looking like
    001.jpg, 002.jpg.
    e.g 1.jpg,...,10.jpg will probably cause some damages.
    """
    frames = []
    for file in sorted(os.listdir(images_dir)):
        frames.append(cv2.imread(os.path.join(images_dir, file)))
    return frames


def read_gt(filename):
    """Read gt and create list of bb-s"""
    assert os.path.exists(filename)
    with open(filename, 'r') as file:
        lines = file.readlines()
    # truncate data (last columns are not needed)
    return [list(map(lambda x: int(x), line.split(',')[:6])) for line in lines]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    This function aims to remove noise from the foreground extracted.
    It is used because the background subtractor detects moving pixels and by doing so we hope
    that it will create homogeneous blocks of pixels that could represent someone rather than
    just having a cloud of pixels detected.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    # threshold
    dilation[dilation < 240] = 0
    return dilation


def extend_contour(x: int, y: int, dx: int, dy: int, frame: np.ndarray) -> Tuple:
    """
    Given a contour this function add some pixels around the contour.
    It aims to extend contours because background subtractor often does not detect whole bodies
    but only some parts.
    """
    pixels_to_add = 10
    xmin = int(max(x - pixels_to_add, 0))
    ymin = int(max(y - pixels_to_add, 0))
    xmax = int(min(frame.shape[1], x + dx + pixels_to_add))
    ymax = int(min(frame.shape[0], y + dy + pixels_to_add))
    return xmin, ymin, xmax, ymax


def are_overlapping(rectangle1: Tuple, rectangle2: Tuple) -> bool:
    """Given two contours, it returns True if they are overlapping"""
    x_overlap = (rectangle1[0] <= rectangle2[0] <= rectangle1[2]) or \
                (rectangle1[0] <= rectangle2[2] <= rectangle1[2]) or \
                (rectangle2[0] <= rectangle1[0] <= rectangle2[2]) or \
                (rectangle2[0] <= rectangle1[2] <= rectangle2[2])
    y_overlap = (rectangle1[1] <= rectangle2[1] <= rectangle1[3]) or \
                (rectangle1[1] <= rectangle2[3] <= rectangle1[3]) or \
                (rectangle2[1] <= rectangle1[1] <= rectangle2[3]) or \
                (rectangle2[1] <= rectangle1[3] <= rectangle2[3])
    return x_overlap and y_overlap


def _merge_contour(contour1: Tuple, contour2: Tuple) -> Tuple:
    """Given two contours, it returns the smallest contour that contains both contours"""
    xmin = min(contour1[0], contour2[0])
    ymin = min(contour1[1], contour2[1])
    xmax = max(contour1[2], contour2[2])
    ymax = max(contour1[3], contour2[3])
    return xmin, ymin, xmax, ymax


def merge_contours(contours: List[Tuple]) -> List[Tuple]:
    """ Given a list of contours, it merges overlapping contours. """
    if len(contours) > 0:
        to_visit = sorted(contours)
        merged_contours = [to_visit.pop()]

        while len(to_visit) > 0:
            current_contour = to_visit.pop()
            merged = False
            for i, contour in enumerate(merged_contours):
                if contour is not None:
                    if are_overlapping(contour, current_contour):
                        to_visit.append(_merge_contour(contour, current_contour))
                        merged_contours[i] = None
                        merged = True
                        break
            if not merged:
                merged_contours.append(current_contour)

        merged_contours = [x for x in merged_contours if x is not None]
        return merged_contours
    else:
        return []


hog = cv2.HOGDescriptor()


def is_pedestrian(img: np.ndarray, model: Model) -> int:
    """
    Given a image returns whether there are pedestrians.
    It resizes the image and computes hog features then uses a trained SVM to detect pedestrians
    """
    _W = 64
    _H = 128
    hog_features = hog.compute(cv2.resize(img, (_W, _H))).reshape(1, -1)
    return model.predict(hog_features)[0]


def contain_pedestrians(img: np.ndarray, model: Model) -> bool:
    """
    Given an image it uses a sliding windows at different scales in order to detect pedestrians.
    As soon as it found a pedestrian it returns True, otherwise it returns False.
    """
    # This is used to limit the size of the sliding window
    min_height = 40
    min_width = 10

    # Scale factor used to reduce the size of the sliding window
    scale = 0.75

    # It is used to compute the translation step size of the sliding window
    translation_factor = 0.20

    H, W, _ = img.shape
    # Number of different sliding window sizes that will be used to detect pedestrians
    depth_max = 3
    h = H
    w = int(2 * W / 3)
    for depth in range(1, depth_max):
        h_step = int(np.ceil((H - h) / (translation_factor * h)))
        w_step = int(np.ceil((W - w) / (translation_factor * w)))
        if h > min_height and w > min_width:
            for i in range(w_step + 1):
                for j in range(h_step + 1):
                    xmin = int(i * translation_factor * w)
                    xmax = int(min(w + np.ceil(i * translation_factor * w), W - 1))

                    ymin = int(j * translation_factor * h)
                    ymax = int(min(h + np.ceil(j * translation_factor * h), H - 1))
                    # As soon as we found pedestrian we return True
                    if is_pedestrian(img[ymin:ymax, xmin:xmax], model):
                        return True
            w, h = int(scale * w), int(scale * h)
        else:
            break
    return False


def pedestrians(data_root: str, _W: int, _H: int, _N: int , model_path='./svm.pickle',
                debug=False):
    """
    Given a folder that contains the frames of a video with the following name 001.jpg, 
    002.jpg...... it returns a list of bounding boxes that contain pedestrians.
    Each element of the list has the following format:
    (frame_id, counter, xmin, ymin, dx, dy)
    - frame_id is the id of the frame
    - counter is a counter that indicates that it's the i-th bounding box detected for the frame 
    frame_id

    :param data_root: folder that contains the frames
    :param _W: not used but required for the assignment
    :param _H: not used but required for the assignment
    :param _N: not used but required for the assignment
    :param model_path: path to the trained SVM for that task
    :param debug: used to debug / visualize in live the detected bounding boxes 
    :return: 
    """

    # Load pretrained svm to detect pedestrians
    model = Model.load_model(model_path)

    frames = read_frames(data_root)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    bboxes = []
    n = len(frames)

    for i, frame in enumerate(frames):
        cpt = 1
        frame = frame.copy()
        # First extract the foreground to capture motion
        fgmask = fgbg.apply(frame)

        # Then we clean the foreground in order to make it more consistent.
        fgmask = clean_mask(fgmask)

        # From the clean foreground we extract bounding boxes where have detected motion
        extended_contours = []
        im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_TC89_L1)

        for c in contours:
            # We extend contours by adding some pixels because sometimes some part of bodies
            # are moving so the whole body is not considered as moving.
            xmin, ymin, xmax, ymax = extend_contour(*cv2.boundingRect(c), frame)
            extended_contours.append((xmin, ymin, xmax, ymax))

        # Then we merge overlapping contours this allow us to merge different part of the body
        # that have been detected separately
        for (xmin, ymin, xmax, ymax) in merge_contours(extended_contours):

            # countour image extraction
            contour_image = frame[ymin:ymax, xmin:xmax].copy()
            # To remove non pedestrian object that are also moving we use a svm trained on
            # recognizing pedestrians
            # and a sliding window
            if contain_pedestrians(contour_image, model):
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                bboxes.append((i, cpt, xmin, ymin, xmax - xmin, ymax - ymin))
                cpt += 1
            else:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

        if debug:
            cv2.imshow('Background Subtraction', fgmask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Frame', frame)
            if i+1 % 50 == 0:
                print(f'{i+1}/{n} frames')
    if debug:
        cv2.destroyAllWindows()
    return bboxes
