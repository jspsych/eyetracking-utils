pip install mediapipe
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
