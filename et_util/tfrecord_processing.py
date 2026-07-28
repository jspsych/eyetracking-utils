import fnmatch
import os
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np

# Import the new MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Note: Ensure this module is available in your workspace
from process_functions import getRightEye, getLeftEye, getBothEyesAsSingleImage


def process_jpg_to_tfr(
        in_path: str,
        out_path: str,
        process,
        model_path="face_landmarker.task",
        overwrite=True,
        verbose=True):
    """Processes jpeg files in a directory to tfrecord files
    in a specified directory

    :param in_path: directory of jpeg files
    :param out_path: directory where tfrecord files will go
    :param process: process helper function
    :param model_path: path to the MediaPipe FaceLandmarker model file (.task)
    """
    error = False
    all_files = os.listdir(in_path)

    # Initialize the new MediaPipe Tasks API Face Landmarker
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

    unique_subjects = set([fname.rsplit('.', 1)[0].split('_')[0] for fname in all_files if fname.endswith('.jpg')])
    files_arr = [fnmatch.filter(all_files, f"{subject}*.jpg") for subject in unique_subjects]

    for subject_files in files_arr:
        if not subject_files:
            continue
            
        subject = subject_files[0].rsplit('.', 1)[0].split("_")[0]
        out_file = os.path.join(out_path, f"{subject}.tfrecords")

        if overwrite and os.path.exists(out_file):
            os.remove(out_file)
            if verbose:
                print(f"Overwriting {out_file}")
        elif not overwrite and os.path.exists(out_file):
            print(f"{out_file} already exists")
            continue

        with tf.io.TFRecordWriter(out_file) as writer:
            for fname in subject_files:
                finfo = fname.rsplit('.', 1)[0].split("_")
                image_path = os.path.join(in_path, fname)

                subject_id = finfo[0]
                if len(finfo) == 3:
                    x, y = finfo[1], finfo[2]
                    tag = {
                        'subject_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(subject_id, 36)])),
                        'x': tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)])),
                        'y': tf.train.Feature(float_list=tf.train.FloatList(value=[float(y)]))
                    }
                else:
                    phase, x, y = finfo[1], finfo[2], finfo[3]
                    tag = {
                        'subject_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(subject_id, 36)])),
                        'x': tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)])),
                        'y': tf.train.Feature(float_list=tf.train.FloatList(value=[float(y)])),
                        'phase': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(phase)]))
                    }
                
                data = process(image_path, face_mesh)
                
                # Verify error safely
                if isinstance(data, dict) and 'error' in data:
                    error = True
                elif data == 'error': # Fallback just in case
                    error = True

                if verbose:
                    print(f"Processed point [{x}, {y}]")
                    if error:
                        print("Above point has bad data, discarding.")

                if error:
                    error = False
                    continue

                tag.update(data)
                example = tf.train.Example(features=tf.train.Features(feature=tag))

                writer.write(example.SerializeToString())
            if verbose:
                print("Generated " + subject_id + ".tfrecords")
            writer.close()


def make_single_example_mediapipe(image_path, face_mesh):
    """Helper process function for process_jpg_to_tfr that 
    defines an example with mediapipe facemesh landmarks
    
    :param image_path: path of image file
    :param face_mesh: mediapipe FaceLandmarker that generates landmarks for face
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image")
        return {'error': True}

    # MediaPipe Tasks require RGB images wrapped in mp.Image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = face_mesh.detect(mp_image)
    
    if not results.face_landmarks:
        print("Cannot make mesh")
        return {'error': True}

    landmarks = results.face_landmarks[0]
    lm_arr = [[l.x, l.y, l.z] for l in landmarks]
    lm_arr_tensor = tf.io.serialize_tensor(lm_arr)

    return {'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr_tensor.numpy()]))}


def make_single_example_jpg(image_path, face_mesh):
    """
    Converts a directory of jpg files to a directory of TFRecord files with one file per unique subject.
    In addition to subject id and labels, TFRecord files include image width,
    image height, and raw image array.

    :param image_path: directory of jpeg files
    :param face_mesh: empty variable needed to integrate with process_jpg_to_tfr
    """
    image = tf.io.read_file(image_path)
    image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int64, name=None)

    feature_description = {
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[1])])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[0])])),
        'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()]))
    }
    return feature_description


def make_single_example_landmarks_and_jpg(image_path, face_mesh):
    """
    Converts jpg file to a dictionary to be used in process_jpg_to_tfr. In addition to
    subject id and labels, TFRecord files include jpg width and height, and raw image array.
    Also includes mediapipe landmarks.

    feature_description = {landmarks, width, height, raw_image}

    :param image_path: directory of jpeg files
    :param face_mesh: mediapipe facemesh
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image")
        return {'error': True}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = face_mesh.detect(mp_image)
    
    if not results.face_landmarks:
        print("Cannot make mesh")
        return {'error': True}

    landmarks = results.face_landmarks[0]
    lm_arr = [[l.x, l.y, l.z] for l in landmarks]
    
    image_tf = tf.io.read_file(image_path)
    lm_arr_tensor = tf.io.serialize_tensor(lm_arr)
    image_shape = tf.io.extract_jpeg_shape(image_tf, output_type=tf.dtypes.int64, name=None)

    feature_description = {
        'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr_tensor.numpy()])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[1])])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[0])])),
        'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_tf.numpy()]))
    }
    return feature_description


def make_single_example_landmarks_and_eyes(image_path, face_mesh):
    """
    Converts jpg file to a dictionary to be used in process_jpg_to_tfr.
    In addition to subject id and labels, TFRecord files include eye image widths and heights,
    and raw left and right eye grayscale image arrays. Also includes mediapipe landmarks.

    feature_description = {landmarks, left_width, right_width, left_height, right_height, left_eye, right_eye}

    :param image_path: directory of jpeg files
    :param face_mesh: mediapipe facemesh
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image")
        return {'error': True}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = face_mesh.detect(mp_image)
    
    if not results.face_landmarks:
        print("Cannot make mesh")
        return {'error': True}

    landmarks = results.face_landmarks[0]
    lm_arr = [[l.x, l.y, l.z] for l in landmarks]

    left_eye_arr = getLeftEye(image, lm_arr)
    if 0 in left_eye_arr.shape:
        print("Cannot get left eye")
        return {'error': True}
    left_eye_arr_gs = cv2.cvtColor(left_eye_arr, cv2.COLOR_BGR2GRAY)
    resized_left = cv2.resize(left_eye_arr_gs, (60, 30))

    right_eye_arr = getRightEye(image, lm_arr)
    if 0 in right_eye_arr.shape:
        print("Cannot get right eye")
        return {'error': True}
    right_eye_arr_gs = cv2.cvtColor(right_eye_arr, cv2.COLOR_BGR2GRAY)
    resized_right = cv2.resize(right_eye_arr_gs, (60, 30))

    left_eye = tf.io.serialize_tensor(resized_left)
    right_eye = tf.io.serialize_tensor(resized_right)
    lm_arr_tensor = tf.io.serialize_tensor(lm_arr)

    feature_description = {
        'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr_tensor.numpy()])),
        'left_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[60])),
        'right_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[60])),
        'left_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[30])),
        'right_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[30])),
        'left_eye': tf.train.Feature(bytes_list=tf.train.BytesList(value=[left_eye.numpy()])),
        'right_eye': tf.train.Feature(bytes_list=tf.train.BytesList(value=[right_eye.numpy()]))
    }
    return feature_description

def make_single_example_landmarks_and_joint_eyes(image_path, face_mesh):
    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image")
        return {'error': True}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = face_mesh.detect(mp_image)
    
    if not results.face_landmarks:
        print("Cannot make mesh")
        return {'error': True}

    landmarks = results.face_landmarks[0]
    lm_arr = [[l.x, l.y, l.z] for l in landmarks]

    eyes_arr = getBothEyesAsSingleImage(image, lm_arr)
    if 0 in eyes_arr.shape:
        print("Cannot get eyes")
        return {'error': True}
        
    eyes_gs = cv2.cvtColor(eyes_arr, cv2.COLOR_BGR2GRAY)
    resized_eyes = cv2.resize(eyes_gs, (144, 36))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    recolored_eyes = clahe.apply(resized_eyes)

    eyes = tf.io.serialize_tensor(recolored_eyes)
    lm_arr_tensor = tf.io.serialize_tensor(lm_arr)

    feature_description = {
        'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr_tensor.numpy()])),
        'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[144])),
        'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[36])),
        'eye_img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[eyes.numpy()]))
    }

    return feature_description

def remove_subject_tfrecords(directory, subject_ids):
    """
    Function that removes TFRecords files based on a list of subject ids.

    :param directory: directory with TFRecord files to be filtered
    :param subject_ids: list of subject ids
    """
    filenames = [subject_id + '.tfrecords' for subject_id in subject_ids]
    for filename in os.listdir(directory):
        if filename in filenames:
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Removed file: {file_path}")


# ==========================================
# TEST BLOCK 
# ==========================================
if __name__ == "__main__":
    import urllib.request
    import shutil

    print("--- Running MediaPipe TFRecord Test Block ---")
    test_in = "test_data_in"
    test_out = "test_data_out"
    os.makedirs(test_in, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    
    # The new API requires a model payload (.task) file to be stored locally
    model_name = "face_landmarker.task"
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_name)
    
    # Create a dummy image (black screen)
    # Note: Using an empty image will result in "Cannot make mesh" since it lacks a face. 
    # It demonstrates that MediaPipe properly skips bad data based on the updated logic.
    dummy_img_path = os.path.join(test_in, "testsub_100_200.jpg")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(dummy_img_path, dummy_img)
    
    print("\n[+] Testing process_jpg_to_tfr...")
    process_jpg_to_tfr(
        in_path=test_in, 
        out_path=test_out, 
        process=make_single_example_mediapipe, 
        model_path=model_name,
        overwrite=True, 
        verbose=True
    )
    
    print("\n[+] Cleaning up test environment...")
    shutil.rmtree(test_in)
    shutil.rmtree(test_out)
    print("Test finished successfully!")
