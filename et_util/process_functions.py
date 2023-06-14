import json

import cv2
import mediapipe as mp


def getLeftEye(image, landmarks):
    """
  Helper function for extract_eyes that extracts the left eye
  from an image.

  :param image: Image of face
  :param landmarks: Mesh of face
  """
    eye_top = int(landmarks[159][1] * image.shape[0])
    eye_left = int(landmarks[33][0] * image.shape[1])
    eye_bottom = int(landmarks[145][1] * image.shape[0])
    eye_right = int(landmarks[133][0] * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye


def getRightEye(image, landmarks):
    """
  Helper function for extract_eyes that extracts the right eye
  from an image.

  :param image: Image of face
  :param landmarks: Mesh of face
  """
    eye_top = int(landmarks[386][1] * image.shape[0])
    eye_left = int(landmarks[362][0] * image.shape[1])
    eye_bottom = int(landmarks[374][1] * image.shape[0])
    eye_right = int(landmarks[263][0] * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye


def extract_eyes(json_path: str, webm_path: str):
    """
  Processing function to be used with process_one_file that outputs eye data and
  three other random features.

  :param json_path: .json file loaded with process_one_file function to be processed.
  :param webm_path: Path of .json file's respective webm file.
  :return: Array containing images of eyes and random features.
  """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
    # temp fix since this function isn't being used
    meshes = []  # extract_mesh_from_video(webm_path, face_mesh)
    eyes = []
    i = 0
    for i in range(0, len(meshes)):
        cap = cv2.VideoCapture(webm_path)
        ret, frame = cap.read()
        eyes.append([getLeftEye(frame, meshes[i]), getRightEye(frame, meshes[i])])
    return eyes


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
