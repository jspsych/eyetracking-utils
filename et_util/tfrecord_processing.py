!pip install mediapipe
import tensorflow as tf
import os
import cv2
import fnmatch
import mediapipe as mp

def extract_meshes_to_tfrecords(
        in_path: str, 
        out_path: str,
        overwrite=True,
        verbose=True):
    """Processes jpeg files in a directory to tfrecord files
    in a specified out path

    :param in_path: directory of jpeg files
    :param out_path: directory where tfrecord files will go
    """
    all_files = os.listdir(in_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)


    unique_subjects = set([fname.rsplit('.', 1)[0].split('_')[0] for fname in all_files])
    files_arr = [fnmatch.filter(all_files, f"{subject}*.jpg") for subject in unique_subjects]

    for subject_files in files_arr:
        subject = subject_files[0].rsplit('.',1)[0].split("_")[0]
        out_file = os.path.join(out_path, f"{subject}.tfrecords")
        with tf.io.TFRecordWriter(out_file) as writer:
            for fname in subject_files:
                finfo = fname.rsplit('.', 1)[0].split("_")
                image_path = os.path.join(in_path, fname)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"{fname} cannot be read")
                    continue 

                results = face_mesh.process(image)
                if not results.multi_face_landmarks:
                    continue

                landmarks = results.multi_face_landmarks[0].landmark
                lm_arr = [[l.x, l.y, l.z] for l in landmarks]
                lm_arr = tf.io.serialize_tensor(lm_arr)

                x, y = finfo[1], finfo[2]

                data = {
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)])),
                    'y': tf.train.Feature(float_list=tf.train.FloatList(value=[float(y)])),
                    'landmarks': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lm_arr.numpy()]))
                }

                example = tf.train.Example(features=tf.train.Features(feature=data))

                """if overwrite and os.path.exists(out_file):
                    os.remove(out_file)
                    if verbose:
                        print(f"Overwriting {out_file}")
                elif not overwrite and os.path.exists(out_file):
                    print(f"{out_file} already exists")
                    continue"""

                writer.write(example.SerializeToString())

                if verbose:
                    print(f"Processed point [{x}, {y}]")
            writer.close()
       

def parse_filename(filename):
  """Helper function for write_images_to_tf_records that parses the filename 
  into subject ID, x-coordinate, and y-coordinate. 

  :param filename: Path to an image file. 
  """
  
  splitname = os.path.splitext(os.path.basename(filename))[0].rsplit('.', 1)[0].split("_")
  subject_id = splitname[0]
  x = float(splitname[1])
  y = float(splitname[2])
  return { "subject_id": subject_id, "x": x, "y": y}

def filter_files(all_files):
  """Helper function for write_images_to_tf_records that filters out a list of unwanted subjects. 
  Currently used to filter out subjects that do not have 480x640 video resolutions.

  :param all_files: List of file names. 
  """

  # list of excluded subjects (from R script)
  exclude_subjects = ["292x6vxf", "65vwsgbt", "9l9gtnv6", "bqmax27b", "p4h9ljon", "vu8fuh1d", "w2vypg1l"]

  filtered_list = [file for file in all_files if not any(partial_name in file for partial_name in exclude_subjects)]

  return filtered_list

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is a tensor
        value = value.numpy()  # get the value of the tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_images_to_tfrecords(in_path: str,
                              out_path: str):
  """Converts a directory of jpeg files to a directory of TFRecord files with one TFRecord per unique subject id. Each TFRecord contains features extracted from filename. 

  :param in_path: directory of jpeg files
  :param out_path: directory where tfrecord files will go
  """

  # create list of jpg files
  all_files = os.listdir(in_path)

  # filter any unwanted subjects
  filtered_files = filter_files(all_files)
  
  # Dictionary to store TFRecord writers by subject ID
  subject_records = {}  
  
  for filename in filtered_files:

      file_features = parse_filename(os.path.join(in_path, filename))
      subject_id = file_features['subject_id']

      # Create a new TFRecord writer if subject ID is new
      if subject_id not in subject_records:
          tfrecord_path = os.path.join(out_path, f"{subject_id}.tfrecords")
          subject_records[subject_id] = tf.io.TFRecordWriter(tfrecord_path)

      # read the JPEG source file into a tf.string
      image = tf.io.read_file(os.path.join(in_path, filename))

      # get the shape of the image from the JPEG file header
      image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int64, name=None)

      # Feature description of image to use when parsing
      feature_description = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[0]),
        'subject_id': _int64_feature(int(file_features['subject_id'], 36)),
        'raw_image': _bytes_feature(image),
        'x': _float_feature(file_features['x']),
        'y': _float_feature(file_features['y'])
      }

      # Parse image using data structure
      example = tf.train.Example(features=tf.train.Features(feature=feature_description))

      # Write file to TFRecord file with matching subject ID
      subject_records[subject_id].write(example.SerializeToString())

  # Close all the TFRecord writers
  for writer in subject_records.values():
      writer.close()

  print(f"Wrote {len(filtered_files)} images to TFRecords")
