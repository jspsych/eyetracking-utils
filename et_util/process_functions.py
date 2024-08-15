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

def getBothEyesAsSingleImage(image, landmarks, padding=0.01):

    left_edge = int((landmarks[226][0] - padding) * image.shape[1])
    right_edge = int((landmarks[446][0] + padding) * image.shape[1])
    
    left_top = int((landmarks[27][1] - padding) * image.shape[0])
    left_bottom = int((landmarks[23][1] + padding) * image.shape[0])

    right_top = int((landmarks[257][1] - padding) * image.shape[0])
    right_bottom = int((landmarks[253][1] + padding) * image.shape[0])

    if(left_top > right_top):
        top = right_top
    else:
        top = left_top
    
    if(left_bottom < right_bottom):
        bottom = right_bottom
    else:
        bottom = left_bottom

    both_eyes = image[top:bottom, left_edge:right_edge]
    return both_eyes

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
