import json
import cv2
import os


def getLeftEye(image, landmarks):
    """
  Helper function for extract_eyes that extracts the left eye
  from an image.

  :param image: Image of face
  :param landmarks: Mesh of face
  """
    eye_top = int(landmarks[27][1] * image.shape[0])
    eye_left = int(landmarks[226][0] * image.shape[1])
    eye_bottom = int(landmarks[23][1] * image.shape[0])
    eye_right = int(landmarks[244][0] * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye


def getRightEye(image, landmarks):
    """
  Helper function for extract_eyes that extracts the right eye
  from an image.

  :param image: Image of face
  :param landmarks: Mesh of face
  """
    eye_top = int(landmarks[257][1] * image.shape[0])
    eye_left = int(landmarks[464][0] * image.shape[1])
    eye_bottom = int(landmarks[253][1] * image.shape[0])
    eye_right = int(landmarks[446][0] * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye


def process_one_file_webm(webm_path: str, in_path: str, file_name: str, process):
    """
    A helper function that opens and processes a file with a specified
    processing function. Modified to accept webm_path as parameter.

    :param webm_path: Path of webm file
    :param in_path: the directory of the .json file
    :param file_name: the name of the file to be processed
    :param process: the processing function
    :return: The dataset returned by the processing function
    """

    with open(in_path + 'c4g2mw61.json') as file:
        data = json.load(file)
        j = data[file_name.split('.')[0]]
    return process(j, webm_path)


def get_last_valid_frame(file_path: str, face_mesh):
    cap = cv2.VideoCapture(file_path)
    frames = []

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
            print(os.path.basename(file_path) + " has no possible frames.")
            return -1
        if results.multi_face_landmarks is not None:
            return results.multi_face_landmarks[0].landmark
        else:
            i -= 1
            print(os.path.basename(file_path) + " has a bad last frame. [Try " + str(-i) + "]")
