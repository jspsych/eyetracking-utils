import numpy as np
import cv2
import os
import fnmatch
import json
import mediapipe as mp

from et_util.process_functions import getRightEye, getLeftEye


def extract_mesh_from_video(path: str, mesh):
    cap = cv2.VideoCapture(path)
    out = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = mesh.process(frame)
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
    return out


def process_webm_to_json(
        in_path: str,
        out_path: str,
        overwrite=True,
        verbose=True):
    all_files = os.listdir(in_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    unique_subjects = set([fname.split('_')[0] for fname in all_files])
    for subject in unique_subjects:
        out_file = os.path.join(out_path, subject + '.json')
        if overwrite and os.path.exists(out_file):
            os.remove(out_file)
            if verbose: print("Overwriting " + out_path)
        if not overwrite and os.path.exists(out_file):
            print(out_path + " already exists")
            continue

        all_data = {}
        subject_data = []
        subject_files = fnmatch.filter(all_files, subject + "*.webm")

        for fname in subject_files:
            finfo = fname.replace('.', '_').split('_')
            subject = finfo[0]
            block = finfo[1]
            phase = finfo[2]
            x = finfo[3]
            y = finfo[4]
            mesh_features = extract_mesh_from_video(in_path + fname, face_mesh)
            subject_data.append({
                'block': block,
                'phase': phase,
                'x': x,
                'y': y,
                'features': mesh_features
            })
            if verbose: print("Processed point [" + x + ", " + y + "]")

        all_data[subject] = subject_data

        with open(out_file, 'w') as file:
            json.dump(all_data, file)
            if verbose:
                print("Generated " + out_file)


def process_webm_to_json_modified(
        in_path: str,
        out_path: str,
        overwrite=True,
        verbose=True):
    """ Modified process_webm_to_json function that includes images of the left
    and right eyes in the json data
    """
    all_files = os.listdir(in_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    unique_subjects = set([fname.split('_')[0] for fname in all_files])
    for subject in unique_subjects:
        out_file = os.path.join(out_path, subject + '.json')
        if overwrite and os.path.exists(out_file):
            os.remove(out_file)
            if verbose: print("Overwriting " + out_path)
        if not overwrite and os.path.exists(out_file):
            print(out_path + " already exists")
            continue

        all_data = {}
        subject_data = []
        subject_files = fnmatch.filter(all_files, subject + "*.webm")

        for fname in subject_files:
            video = in_path + fname
            finfo = fname.replace('.', '_').split('_')
            subject = finfo[0]
            block = finfo[1]
            phase = finfo[2]
            x = finfo[3]
            y = finfo[4]
            mesh_features = extract_mesh_from_video(video, face_mesh)
            left = []
            right = []
            for i in range(0, len(mesh_features)):
                cap = cv2.VideoCapture(video)
                ret, frame = cap.read()
                left.append(getLeftEye(frame, mesh_features[i]))
                right.append(getRightEye(frame, mesh_features[i]))
            subject_data.append({
                'block': block,
                'phase': phase,
                'x': x,
                'y': y,
                'features': mesh_features,
                'left_eyes': left,
                'right_eyes': right
            })
            if verbose: print("Processed point [" + x + ", " + y + "]")

        all_data[subject] = subject_data
        dumped = json.dumps(all_data, cls=NumpyEncoder)

        with open(out_file, 'w') as file:
            json.dump(dumped, file)
            if verbose:
                print("Generated " + out_file)
