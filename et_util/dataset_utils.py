import json
import os

import tensorflow as tf


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
    assert ((train_size >= 1) and (val_size >= 1)) and (test_size >= 1)
    if verbose:
        print(f'Training size: {train_size}')
        print(f'Validation size: {val_size}')
        print(f'Test size: {test_size}')

    # train partition
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

    # val partition
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

    # test partition
    first_file = all_files.pop(0)
    test_ds = process_one_file(in_path, first_file, process)
    test_size -= 1
    for i in range(test_size):
        file = all_files.pop(0)
        test_ds = test_ds.concatenate(process_one_file(in_path, file, process))

    return train_ds, val_ds, test_ds


def process_one_file(in_path: str, file_name: str, process):
    """
    A helper function that opens and processes a file with a specified
    processing function.

    :param in_path: the directory of the .json file
    :param file_name: the name of the file to be processed
    :param process: the processing function
    :return: The dataset returned by the processing function
    """
    with open(in_path + file_name,) as file:
        data = json.load(file)
        j = data[file_name.split('.')[0]]

    return process(j)
