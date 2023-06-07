import numpy as np
import cv2
import os
import fnmatch
import json
import mediapipe as mp

from et_util.process_functions import getRightEye, getLeftEye


def process_webm_to_json(
        in_path: str,
        out_path: str,
        process,
        overwrite=True,
        verbose=True):
    """
    Processes a directory of .webms into .jsons with a given process function.

    :param in_path: the directory containing the .webm files
    :param out_path: the directory where the .json files will be written to
    :param process: the processing function taking in the path of the .webm and the face mesh,
    outputting a json object of the data extracted from the .webm
    :param overwrite: True if the files in the output directory should be overwritten
    :param verbose: True if print statements showing the processed points should be displayed
    """
    error = False
    all_files = os.listdir(in_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    unique_subjects = set([fname.split('_')[0] for fname in all_files])
    for subject in unique_subjects:
        out_file = os.path.join(out_path, subject + '.json')
        if overwrite and os.path.exists(out_file):
            os.remove(out_file)
            if verbose: print("Overwriting " + out_file)
        if not overwrite and os.path.exists(out_file):
            print(out_file + " already exists")
            continue

        all_data = {}
        subject_data = []
        subject_files = fnmatch.filter(all_files, subject + "*.webm")

        for fname in subject_files:
            finfo = os.path.splitext(fname)[0].split('_')
            subject = finfo[0]
            x = finfo[1]
            y = finfo[2]
            tag_json = {
                'x': x,
                'y': y
            }
            data_json = process(in_path + fname, face_mesh)
            if 'error' in data_json:
                error = True
            tag_json.update(data_json)
            subject_data.append(tag_json)
            if verbose:
                print("Processed point [" + x + ", " + y + "]")

        if not error:
            all_data[subject] = subject_data
            with open(out_file, 'w') as file:
                json.dump(all_data, file)
            if verbose:
                print("Generated " + out_file)
        if error and verbose:
            print("Above point has bad data, discarding.")
        error = False


def get_landmarks(path, face_mesh):
    """
    A process function that gets just MediaPipe facial landmarks.

    :param path: the path of the .webm video
    :param face_mesh: the MediaPipe face mesh
    :return: a .json containing an array with each frame's facial landmarks
    """
    cap = cv2.VideoCapture(path)
    out = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            results = face_mesh.process(frame)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            lm_arr = [[lm.x, lm.y, lm.z] for lm in landmarks]
            out.append(lm_arr)
        else:
            break

    size = len(out)
    if size > 0:
        out = np.reshape(np.array(out), (size, -1, 3)).tolist()
    return {"landmarks": out}


def get_landmarks_and_eyedata(path, face_mesh):
    """
    A process function that gets MediaPipe facial landmarks and the image data
    of the left and right eye.

    :param path: the path of the .webm video
    :param face_mesh: the MediaPipe face mesh
    :return: a .json containing an array with each frame's facial landmarks and eye image data
    """
    cap = cv2.VideoCapture(path)
    out = []
    left = []
    right = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            results = face_mesh.process(frame)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            lm_arr = [[lm.x, lm.y, lm.z] for lm in landmarks]
            out.append(lm_arr)
            left.append(getLeftEye(frame, lm_arr).tolist())
            right.append(getRightEye(frame, lm_arr).tolist())
        else:
            break
    size = len(out)
    if size > 0:
        out = np.reshape(np.array(out), (size, -1, 3)).tolist()

    return {
        "landmarks": out,
        "left": left,
        "right": right
    }


def get_everything(path, face_mesh):
    """
    A process function that gets the MediaPipe facial landmarks and the whole frame image data.

    :param path: the path of the .webm video
    :param face_mesh: the MediaPipe face mesh
    :return: a .json containing an array with each frame's facial landmarks and frame image data.
    """
    cap = cv2.VideoCapture(path)
    out = []
    image = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            results = face_mesh.process(frame)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            lm_arr = [[lm.x, lm.y, lm.z] for lm in landmarks]
            out.append(lm_arr)
            image.append(frame.tolist())
        else:
            break
    size = len(out)
    if size > 0:
        out = np.reshape(np.array(out), (size, -1, 3)).tolist()

    return {
        "landmarks": out,
        "image": image
    }


def get_everything_last_frame(path, face_mesh):
    """
    A process function that gets the last frame's MediaPipe facial landmarks and whole image data.

    :param path: the path of the .webm video
    :param face_mesh: the MediaPipe face mesh
    :return: a .json containing an array with the last frame's facial landmarks and frame image data.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    out = []
    image = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    i = -1
    while True:
        try:
            results = face_mesh.process(frames[i])
        except:
            print(path + " has no possible frames.")
            return {
                "landmarks": "NO FACES DETECTED",
                "image": "NO FACES DETECTED"
            }
        if results.multi_face_landmarks is not None:
            break
        else:
            i -= 1
            print(path + " has a bad last frame. [Try " + str(-i) + "]")

    landmarks = results.multi_face_landmarks[0].landmark
    lm_arr = [[lm.x, lm.y, lm.z] for lm in landmarks]
    out.append(lm_arr)
    image.append(frames[-1].tolist())

    size = len(out)
    if size > 0:
        out = np.reshape(np.array(out), (size, -1, 3)).tolist()
    return {
        "landmarks": out,
        "image": image
    }


def get_everything_last_frame_gs(path, face_mesh):
    """
    A process function that gets the last frame's MediaPipe facial landmarks and whole image data in grayscale.

    :param path: the path of the .webm video
    :param face_mesh: the MediaPipe face mesh
    :return: a .json containing an array with the last frame's facial landmarks and grayscale frame image data.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    out = []
    image = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    i = -1
    while True:
        try:
            results = face_mesh.process(frames[i])
        except:
            print(path + " has no possible frames.")
            return {
                'error': 1
            }
        if results.multi_face_landmarks is not None:
            break
        else:
            i -= 1
            print(path + " has a bad last frame. [Try " + str(-i) + "]")

    landmarks = results.multi_face_landmarks[0].landmark
    lm_arr = [[lm.x, lm.y, lm.z] for lm in landmarks]
    out.append(lm_arr)

    last_image = frames[-1]
    gs_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
    image.append(gs_image.tolist())

    size = len(out)
    if size > 0:
        out = np.reshape(np.array(out), (size, -1, 3)).tolist()
    return {
        "landmarks": out,
        "image": image
    }
