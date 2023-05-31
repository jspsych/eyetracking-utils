import json
import os

import tensorflow as tf


def generate_single_ds(json_data):
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
    with open(in_path + file_name,) as file:
        data = json.load(file)
        j = data[file_name.split('.')[0]]

    return process(j)
