import tensorflow as tf
import os
import cv2
import fnmatch
import mediapipe as mp


def process_jpg_to_tfr(
        in_path: str, 
        out_path: str,
        process,
        overwrite=True,
        verbose=True):
    """Processes jpeg files in a directory to tfrecord files
    in a specified directory

    :param in_path: directory of jpeg files
    :param out_path: directory where tfrecord files will go
    :param process: process helper function
    """
    error = False
    all_files = os.listdir(in_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    unique_subjects = set([fname.rsplit('.', 1)[0].split('_')[0] for fname in all_files])
    files_arr = [fnmatch.filter(all_files, f"{subject}*.jpg") for subject in unique_subjects]

    for subject_files in files_arr:

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
                x, y = finfo[1], finfo[2]

                tag = {
                    'subject_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(subject_id, 36)])),
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)])),
                    'y': tf.train.Feature(float_list=tf.train.FloatList(value=[float(y)]))
                }

                data = process(image_path, face_mesh)
                if 'error' in data:
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
            error = False
            writer.close()


def make_single_example_mediapipe(image_path, face_mesh):
    """Helper process function for process_jpg_to_tfr that 
    defines an example with mediapipe facemesh landmarks
    
    :param image_path: path of image file
    :param face_mesh: mediapipe face mesh that generates landmarks 
    for face"""

    image = cv2.imread(image_path)
    if image is None:
        print("Cannot read image")
        return 'error'

    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        print("Cannot make mesh")
        return 'error'
    
    landmarks = results.multi_face_landmarks[0].landmark
    lm_arr = [[l.x, l.y, l.z] for l in landmarks]
    lm_arr = tf.io.serialize_tensor(lm_arr)
    
    return {'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr.numpy()]))}
       
def make_single_example_jpg(image_path, face_mesh):
  """Converts a directory of jpg files to a directory of TFRecord files with one file per unique subject. In addition to subject id and labels, TFRecord files include image width, image height, and raw image array.

  :param image_path: directory of jpeg files
  :param face_mesh: empty variable needed to integrate with process_jpg_to_tfr
  """

  # read the JPEG source file into a tf.string
  image = tf.io.read_file(image_path)

  # get the shape of the image from the JPEG file header
  image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int64, name=None)

  # Feature description of image to use when parsing
  feature_description = {
    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[1])])),
    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[0])])),
    'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()]))
  }

  return feature_description

def make_single_example_landmarks_and_jpgs(image_path, face_mesh):
  """Converts jpg file to a dictionary to be used in process_jpg_to_tfr. In addition to subject id and labels, 
  TFRecord files include jpg width and height, and raw image array. Also includes mediapipe landmarks.

  feature_description = {landmarks, width, height, raw_image}

  :param image_path: directory of jpeg files
  :param face_mesh: mediapipe facemesh
  """
  
  image = cv2.imread(image_path)
  if image is None:
        print("Cannot read image")
        return 'error'

  results = face_mesh.process(image)
  if not results.multi_face_landmarks:
      print("Cannot make mesh")
      return 'error'
  
  landmarks = results.multi_face_landmarks[0].landmark
  lm_arr = [[l.x, l.y, l.z] for l in landmarks]

  image = tf.io.read_file(image_path)

  lm_arr = tf.io.serialize_tensor(lm_arr)

  # get the shape of the image from the JPEG file header

  image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int64, name=None)

  # Feature description of image to use when parsing
  feature_description = {
    'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr.numpy()])),
    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[1])])),
    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_shape[0])])),
    'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy()]))
  }
  return feature_description

def remove_subject_tfrecords(directory, subject_ids):
  """Function that removes TFRecords files based on a list of subject ids. 

  :param directory: directory with TFRecord files to be filtered
  :param subject_ids: list of subject ids
  """

  filenames = [subject_id + '.tfrecords' for subject_id in subject_ids]
  for filename in os.listdir(directory):
      if filename in filenames:
          file_path = os.path.join(directory, filename)
          os.remove(file_path)
          print(f"Removed file: {file_path}")
