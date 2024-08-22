import json
import os

import tensorflow as tf


def process_json_to_tfds(in_path: str,
                         process,
                         train_split=0.8,
                         val_split=0.1,
                         test_split=0.1,
                         verbose=True):
    """
    Processes a directory with only .json files into tensorflow datasets corresponding
    to the desired proportions for training, validation, and testing based on number of
    participants.

    The proportions of data must add up to 1 and each dataset must have enough for 1
    participant per dataset.

    :param in_path: the directory containing the .json files
    :param process: the processing function used for turning the json data into a tf dataset
    :param train_split: the proportion of participants used for training data
    :param val_split: the proportion of participants used for validation data
    :param test_split: the proportion of participants used for test data
    :param verbose: True if this function should print statements on which jsons are processed
    :return: A tuple of tf.data.Dataset of the shape (train_ds, val_ds, test_ds)
    """
    assert (train_split + val_split + test_split) == 1

    # gather lengths
    all_files = os.listdir(in_path)
    size = len(all_files)
    train_size = int(size * train_split)
    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size += (size - train_size - val_size - test_size)
    
    if verbose:
        print(f'Training size: {train_size}')
        print(f'Validation size: {val_size}')
        print(f'Test size: {test_size}')

    # train partition
    if train_size > 0:
        first_file = all_files.pop(0)
        train_ds = process_one_file(in_path, first_file, process)
        train_size -= 1
        if verbose:
            print("Training data:")
            print(f'{first_file} processed')
        for i in range(train_size):
            file = all_files.pop(0)
            train_ds = train_ds.concatenate(process_one_file(in_path, file, process))
            if verbose:
                print(f'{file} processed')
    else:
        train_ds = None
    
    # val partition
    if val_size > 0:
        first_file = all_files.pop(0)
        val_ds = process_one_file(in_path, first_file, process)
        val_size -= 1
        if verbose:
            print("Validation data:")
            print(f'{first_file} processed')
        for i in range(val_size):
            file = all_files.pop(0)
            val_ds = val_ds.concatenate(process_one_file(in_path, file, process))
            if verbose:
                print(f'{file} processed')
    else:
        val_ds = None

    # test partition
    if test_size > 0:
        first_file = all_files.pop(0)
        test_ds = process_one_file(in_path, first_file, process)
        test_size -= 1
        if verbose:
            print("Test data:")
            print(f'{first_file} processed')
        for i in range(test_size):
            file = all_files.pop(0)
            test_ds = test_ds.concatenate(process_one_file(in_path, file, process))
            if verbose:
                print(f'{file} processed')
    else:
        test_ds = None

    return train_ds, val_ds, test_ds

def generate_single_ds(json_data):
    """
    Creates a tensorflow Dataset of the format (features, labels) to be used
    for training a model. The features correspond to all landmarks, while the
    labels correspond to the x, y position of the landmarks. This function will
    convert all points in the json data into individual elements.

    :param json_data: json of the shape given by our process_webm_to_json()
    :return: A tf.data.Dataset that can be used for model.fit()
    """
    inputs = []
    labels = []

    for block in json_data:
        for frame in block['features']:
            inputs.append(frame)
            labels.append([int(block['x']), int(block['y'])])

    input_ds = tf.data.Dataset.from_tensor_slices(inputs)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    return tf.data.Dataset.zip((input_ds, label_ds))

def gen_everything_last_frame_gs(json_data):
    """
    Creates a tensorflow Dataset of the format {"images":, "landmarks":}, {"labels":}
    for use in training a model.

    :param json_data: json of the shape given by our process_webm_to_json()
    :return: A tf.data.Dataset that can be used for model.fit()
    """
    landmarks = []
    images = []
    labels = []

    for block in json_data:
        landmarks.append(block['landmarks'][0])
        images.append(block['image'][0])
        labels.append([float(block['x']), float(block['y'])])

    return tf.data.Dataset.from_tensor_slices(
        (
            {"images": images, "landmarks": landmarks}, {"labels": labels}
        )
    )

def gen_eye_data(json_data):
    """
    Creates a tensorflow Dataset of the format {"left":, "right":}, {"labels":}
    for use in training a model.

    :param json_data: json of the shape given by our process_webm_to_json()
    :return: A tf.data.Dataset that can be used for model.fit()
    """
    left = []
    right = []
    labels = []
    for block in json_data:
        left.append(block['left'])
        right.append(block['right'])
        labels.append([float(block['x']), float(block['y'])])

    return tf.data.Dataset.from_tensor_slices(
        (
            {"left": left, "right": right}, {"labels": labels}
        )
    )

def process_one_file(in_path: str, file_name: str, process):
    """
    A helper function that opens and processes a file with a specified
    processing function.

    :param in_path: the directory of the .json file
    :param file_name: the name of the file to be processed
    :param process: the processing function
    :return: The dataset returned by the processing function
    """
    with open(in_path + file_name, ) as file:
        data = json.load(file)
        j = data[file_name.split('.')[0]]

    return process(j)

def process_tfr_to_tfds(directory_path,
                        process,
                        filter_imgs=False,
                        train_split=0.8,
                        val_split=0.1,
                        test_split=0.1,
                        random_seed=None,
                        group_function=lambda le, re, m, c, z: z):
    """
    Creates a parsed tensorflow dataset from a directory of tfrecords
    files.

    :param directory_path: path of directory containing tfrecords files. 
    Make sure to include / at end.
    :param process: process function that corresponds to shape of data
    :param filter_imgs: True if images should all be of shape=(640,480,3), False otherwise
    :return: parsed tensorflow dataset
    """
    assert (train_split + val_split + test_split) == 1

    files_arr = os.listdir(directory_path)
    file_paths = [directory_path + file_name for file_name in files_arr]

    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(process)
    if filter_imgs:
        dataset = dataset.filter(lambda *args: tf.math.reduce_all(tf.math.equal(tf.shape(args[0]), (640, 480, 3))))
        dataset = dataset.map(lambda *args: (tf.reshape(args[0], (640, 480, 3)),) + args[1:])

    dataset_grouped = dataset.group_by_window(
            key_func= group_function,
            reduce_func= lambda key, dataset: dataset.batch(144),
            window_size= 144
        )

    n_groups = 0
    for _ in dataset_grouped.as_numpy_iterator():
        n_groups += 1

    train_size = int(train_split * n_groups)
    val_size = int(val_split * n_groups)

    shuffled_dataset = dataset_grouped.shuffle(n_groups, seed=random_seed, reshuffle_each_iteration=False)

    train_grouped_ds = shuffled_dataset.take(train_size)
    val_grouped_ds = shuffled_dataset.skip(train_size).take(val_size)
    test_grouped_ds = shuffled_dataset.skip(train_size).skip(val_size)

    # Ungroup the datasets
    train_ds = train_grouped_ds.unbatch()
    val_ds = val_grouped_ds.unbatch()
    test_ds = test_grouped_ds.unbatch()

    return train_ds, val_ds, test_ds

def get_subject_id(tfdata):
    """Helper function for process_tfr_to_tfds"""
    subject_id = tfdata.map(lambda *args: args[-1])
    id_int = list(subject_id.as_numpy_iterator())[0]
    return id_int

def parse_tfr_element_mediapipe(element):
    """Process function that parses a tfr element in a raw dataset for 
    process_tfr_to_tfds function.
    Gets subject id as unique integer, x and y coordinates as label and 
    mediapipe landmarks.
    Use for data generated with make_single_example_mediapipe (i.e.
    data in landmarks_tfrecords.tar.gz).
    
    :param element: tfr element in raw dataset
    :return: landmarks, label(x, y), subject_id"""
    data = {
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'landmarks': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    subject_id = content['subject_id']
    label = [content['x'], content['y']]
    landmarks = content['landmarks']

    feature = tf.io.parse_tensor(landmarks, out_type=tf.float32)
    feature = tf.reshape(feature, shape=(478, 3))
    return feature, label, subject_id


def parse_tfr_element_jpg(element):
    """Process function that parses a tfr element in a raw dataset for process_tfr_to_tfds function.
    Gets raw image, image width, image height, subject id, and xy labels.
    Use for data generated with make_single_example_jpg (i.e. data in jpg_tfrecods.tar.gz).

    :param element: tfr element in raw dataset
    :return: raw image with shape
    """

    data_structure = {
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data_structure)

    raw_image = content['raw_image']
    width = content['width']
    height = content['height']
    depth = 3
    label = [content['x'], content['y']]
    subject_id = content['subject_id']

    image = tf.io.decode_jpeg(raw_image)
    image = tf.reshape(image, shape=(width, height, depth))

    return image, label, subject_id


def parse_tfr_element_jpg_and_mediapipe(element):
    """Process function that parses a tfr element in a raw dataset for process_tfr_to_tfds function.
    Gets mediapipe landmarks, raw image, image width, image height, subject id, and xy labels.
    Use for data generated with make_single_example_landmarks_and_jpg (i.e. data in
    jpg_landmarks_tfrecords.tar.gz)

    :param element: tfr element in raw dataset
    :return: image, label(x,y), landmarks, subject_id
    """

    data_structure = {
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data_structure)

    landmarks = content['landmarks']
    raw_image = content['raw_image']
    width = content['width']
    height = content['height']
    depth = 3
    label = [content['x'], content['y']]
    subject_id = content['subject_id']

    landmarks = tf.io.parse_tensor(landmarks, out_type=tf.float32)
    landmarks = tf.reshape(landmarks, shape=(478, 3))

    image = tf.io.decode_jpeg(raw_image)
    image = tf.reshape(image, shape=(width, height, depth))

    return image, landmarks, label, subject_id


def parse_tfr_element_eyes_and_mediapipe(element):
    """Process function that parses a tfr element in a raw dataset for process_tfr_to_tfds function.
    Gets mediapipe landmarks, raw eye images, eye image width, eye image height, subject id, and xy labels.
    Eye images are reshaped to (60, 30).
    Use for data generated with make_single_example_landmarks_and_eyes (i.e. data in
    eyes_landmarks_tfrecords.tar.gz).

    :param element: tfr element in raw dataset
    :return: left_eye, right_eye, landmarks, label(x,y), subject_id
    """

    data_structure = {
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'left_width': tf.io.FixedLenFeature([], tf.int64),
        'right_width': tf.io.FixedLenFeature([], tf.int64),
        'left_height': tf.io.FixedLenFeature([], tf.int64),
        'right_height': tf.io.FixedLenFeature([], tf.int64),
        'left_eye': tf.io.FixedLenFeature([], tf.string),
        'right_eye': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data_structure)

    subject_id = content['subject_id']
    label = [content['x'], content['y']]
    landmarks = content['landmarks']
    left_width = content['left_width']
    right_width = content['right_width']
    left_height = content['left_height']
    right_height = content['right_height']
    left_eye = content['left_eye']
    right_eye = content['right_eye']

    landmarks = tf.io.parse_tensor(landmarks, out_type=tf.float32)
    landmarks = tf.reshape(landmarks, shape=(478, 3))

    left_eye = tf.io.parse_tensor(left_eye, out_type=tf.uint8)
    left_eye = tf.reshape(left_eye, shape=(left_width, left_height))
    left_eye = tf.reshape(left_eye, shape=(60,30))

    right_eye = tf.io.parse_tensor(right_eye, out_type=tf.uint8)
    right_eye = tf.reshape(right_eye, shape=(right_width, right_height))
    right_eye = tf.reshape(right_eye, shape=(60,30))

    return left_eye, right_eye, landmarks, label, subject_id

def parse_tfr_element_single_eye_image_and_mediapipe(element):
    """Process function that parses a tfr element in a raw dataset for process_tfr_to_tfds function.
    Gets mediapipe landmarks, raw image, image width, image height, subject id, and xy labels.
    

    :param element: tfr element in raw dataset
    :return: image, label(x,y), landmarks, subject_id
    """

    data_structure = {
        'landmarks': tf.io.FixedLenFeature([], tf.string),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenFeature([], tf.float32),
        'y': tf.io.FixedLenFeature([], tf.float32),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'subject_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data_structure)

    landmarks = content['landmarks']
    raw_image = content['raw_image']
    width = content['width']
    height = content['height']
    depth = 3
    label = [content['x'], content['y']]
    subject_id = content['subject_id']

    landmarks = tf.io.parse_tensor(landmarks, out_type=tf.float32)
    landmarks = tf.reshape(landmarks, shape=(478, 3))

    image = tf.io.decode_jpeg(raw_image)
    image = tf.reshape(image, shape=(height, width, depth))

    return image, landmarks, label, subject_id
