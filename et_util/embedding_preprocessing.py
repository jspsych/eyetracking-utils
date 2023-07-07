import tensorflow as tf
import numpy as np
import math
import os

def find_relative_distance(p1, p2, p3):
    # Find the slope of the line formed by p1 and p2
    slope = tf.cond(
        tf.equal(p2[0], p1[0]),
        lambda: float('inf'),
        lambda: (p2[1] - p1[1]) / (p2[0] - p1[0])
    )

    # Find the negative reciprocal of the slope to get the slope of the perpendicular line
    perp_slope = tf.cond(
        tf.equal(slope, 0),
        lambda: float('inf'),
        lambda: -1 / slope
    )

    # Use the point-slope form of a line to find the equation of the perpendicular line
    b = p3[1] - perp_slope * p3[0]

    # Solve the system of equations to find the point of intersection
    x = tf.cond(
        tf.equal(slope, float('inf')),
        lambda: p1[0],
        lambda: tf.cond(
            tf.equal(perp_slope, float('inf')),
            lambda: p3[0],
            lambda: (b - p1[1] + slope * p1[0]) / (slope - perp_slope)
        )
    )
    y = tf.cond(
        tf.equal(slope, float('inf')),
        lambda: perp_slope * x + b,
        lambda: tf.cond(
            tf.equal(perp_slope, float('inf')),
            lambda: slope * x + p1[1] - slope * p1[0],
            lambda: slope * x + p1[1] - slope * p1[0]
        )
    )

    # Compute the distance between the first point and the point of intersection
    # and divides by the distance between first two points
    distance = tf.sqrt(tf.square(x - p1[0]) + tf.square(y - p1[1])) / tf.sqrt(tf.square(p2[0] - p1[0]) + tf.square(p2[1] - p1[1]))

    return distance

def map_to_coordinate_space(x1, x2, y1, y2, points):
    x_vals = tf.map_fn(lambda _x: find_relative_distance(x1, x2, _x), points)
    y_vals = tf.map_fn(lambda _y: find_relative_distance(y1, y2, _y), points)

    return tf.stack([x_vals, y_vals], axis=1)

# Right eye:
#   Anchors Top: 27 Bottom: 23 Left: 130 Right: 243
#   Iris 473-477

# Left eye:
#   Anchors Left: 463 Right: 359 Top: 257 Bottom: 253
#   Iris 468-472

def norm_facemesh(facemesh):
    """ Function to normalize face mesh points. Use for landmarks_tfrecords dataset. 
    :param facemesh: Tensor of mediapipe face landmarks
    :return: A tuple containing right and left eye normalized landmarks"""
    right_eye_x1 = tf.reshape(tf.gather(tf.gather(facemesh, [130], axis=0), [0,1], axis=1), (2,))
    right_eye_x2 = tf.reshape(tf.gather(tf.gather(facemesh, [243], axis=0), [0,1], axis=1), (2,))

    right_eye_y1 = tf.reshape(tf.gather(tf.gather(facemesh, [23], axis=0), [0,1], axis=1), (2,))
    right_eye_y2 = tf.reshape(tf.gather(tf.gather(facemesh, [27], axis=0), [0,1], axis=1), (2,))

    left_eye_x1 = tf.reshape(tf.gather(tf.gather(facemesh, [463], axis=0), [0,1], axis=1), (2,))
    left_eye_x2 = tf.reshape(tf.gather(tf.gather(facemesh, [359], axis=0), [0,1], axis=1), (2,))

    left_eye_y1 = tf.reshape(tf.gather(tf.gather(facemesh, [253], axis=0), [0,1], axis=1), (2,))
    left_eye_y2 = tf.reshape(tf.gather(tf.gather(facemesh, [257], axis=0), [0,1], axis=1), (2,))

    left_eye_points = tf.gather(tf.gather(facemesh, [473, 474, 475, 476, 477], axis=0), [0,1], axis=1)

    right_eye_points = tf.gather(tf.gather(facemesh, [468, 469, 470, 471, 472], axis=0), [0,1], axis=1)

    right_eye_norm = map_to_coordinate_space(right_eye_x1, right_eye_x2, right_eye_y1, right_eye_y2, right_eye_points)
    left_eye_norm = map_to_coordinate_space(left_eye_x1, left_eye_x2, left_eye_y1, left_eye_y2, left_eye_points)

    return (right_eye_norm, left_eye_norm)
    
def group_dataset(dataset, window_size):
    """Groups dataset into groups where the key_func returns the same value
    i.e., the subject_id is the same. reduce_func is applied to each grouped
    dataset. window_size sets the maximum number of dataset elements in a grouped
    dataset.
    :param dataset: dataset to group - pass test, validation, and train data
    individually into the function.
    :param window_size: desired number of dataset elements in grouped dataset"""

    def reduce_func(key, grouped_dataset):
        # drop_remainder is important because we want the batches to have the 
        # desired size and only the desired size. we are using batch() to create 
        # pairs of data in this case, *not* to batch for training the network in 
        # the standard use of batch
        return grouped_dataset.batch(window_size, drop_remainder=True)

    def key_func(*args):
        return args[-1]  
        
    transformed_data = dataset.shuffle(10000).group_by_window(
        key_func,
        reduce_func,
        window_size
    )

    return transformed_data