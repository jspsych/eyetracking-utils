import tensorflow as tf

from et_util.custom_loss import normalized_weighted_euc_dist

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
    :param shuffled_dataset: shuffled dataset to group - pass test, validation, and train data
    individually into the function.
    :param window_size: desired number of dataset elements in grouped dataset
    :return: Grouped dataset"""

    def reduce_func(key, grouped_dataset):
        # drop_remainder is important because we want the batches to have the 
        # desired size and only the desired size. we are using batch() to create 
        # pairs of data in this case, *not* to batch for training the network in 
        # the standard use of batch
        return grouped_dataset.batch(window_size, drop_remainder=True)

    def key_func(*args):
        return args[-1]  
        
    transformed_data = dataset.group_by_window(
        key_func,
        reduce_func,
        window_size
    )

    return transformed_data

def map_norm_func(x,y,z):
    input = norm_facemesh(x)

    return (input, y, z)

def mediapipe_triplet_map_combine_func(x,y,z):
    """Takes coordinates from y -- shape (3,2) -- and calculates
    the distance between them and uses that to output normalized facemesh points
    in a form to be used in triplet loss. Dataset must contain mediapipe landmarks and 
    coordinate labels in format ([other features (optional)], landmarks, 
    label, subject_id).

    :param data_tuple: grouped tuple containing three elements from 
    dataset.
    :return: tuple of format (archor_right, anchor_left, positive_right,
    positive_left, negative_right, negative_left)"""

    point1 = tf.gather(y, [0])
    point2 = tf.gather(y, [1])
    point3 = tf.gather(y, [2])

    dist_1_2 = normalized_weighted_euc_dist(point1, point2)
    dist_1_3 = normalized_weighted_euc_dist(point1, point3)

    # 128 is maximum distance between points in our setup.
    # similarity = tf.constant([1.]) - (dist / tf.constant([128.]))

    (right_eyes, left_eyes) = x

    (anchor_right, anchor_left) = (
        tf.reshape(tf.gather(right_eyes, [0]), (5,2)),
        tf.reshape(tf.gather(left_eyes, [0]), (5,2)))

    (positive_right, positive_left) = tf.cond(tf.less(dist_1_2, dist_1_3),
                        lambda: (
        tf.reshape(tf.gather(right_eyes, [1]), (5,2)),
        tf.reshape(tf.gather(left_eyes, [1]), (5,2))),
                        lambda: (
        tf.reshape(tf.gather(right_eyes, [2]), (5,2)),
        tf.reshape(tf.gather(left_eyes, [2]), (5,2))))

    (negative_right, negative_left) = tf.cond(tf.less(dist_1_2, dist_1_3),
                        lambda: (
        tf.reshape(tf.gather(right_eyes, [2]), (5,2)),
        tf.reshape(tf.gather(left_eyes, [2]), (5,2))),
                        lambda: (
        tf.reshape(tf.gather(right_eyes, [1]), (5,2)),
        tf.reshape(tf.gather(left_eyes, [1]), (5,2))))


    return ((anchor_right, anchor_left, positive_right, positive_left, negative_right, negative_left))

def get_triplet_data_mediapipe(dataset, train=False):
    """Processes dataset with landmarks into form that can be passed into triplet
    loss model. Must batch dataset before passing to model.
    :param dataset: Dataset with shape (landmarks, label, subject_id)
    :param train: If dataset is train dataset, set to True
    :return: Processed dataset with shape ((5,2), (5,2), (5,2), (5,2), (5,2), (5,2))"""
    cached = dataset.map(map_norm_func).cache()
    if train:
        grouped_dataset = group_dataset(
            cached.repeat().shuffle(20000), 
            3).map(mediapipe_triplet_map_combine_func)
    else:
        grouped_dataset = group_dataset(
            cached, 3).map(mediapipe_triplet_map_combine_func)
    return grouped_dataset

class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]
