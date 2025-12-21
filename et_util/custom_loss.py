import keras
from keras import ops


@keras.saving.register_keras_serializable(package="et_util")
def normalized_weighted_euc_dist(y_true, y_pred):
    """Custom loss function that calculates a weighted Euclidean distance between two sets of points.

    Weighting multiplies the x-coordinate of each input by 1.778 (derived from the 16:9 aspect ratio of most laptops) to match the scale of the y-axis.
    For interpretability, the distances are normalized to the diagonal such that the maximum distance between two points (corner to corner) is 100.

    :param y_true (tensor): A tensor of shape (batch, 2) containing ground-truth x- and y- coordinates
    :param y_pred (tensor): A tensor of shape (batch, 2) containing predicted x- and y- coordinates

    :returns: A tensor of shape (batch,) with the weighted and normalized euclidean distances between the points in y_true and y_pred.
    """
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")

    # Weighting treats y-axis as unit-scale and creates a rectangle that's 177.8x100 units.
    x_weight = ops.array([1.778, 1.0], dtype="float32")

    # Multiply x-coordinate by 16/9 = 1.778
    y_true_weighted = ops.multiply(x_weight, y_true)
    y_pred_weighted = ops.multiply(x_weight, y_pred)

    # Calculate Euclidean distance with weighted coordinates
    loss = ops.sqrt(ops.sum(ops.square(y_pred_weighted - y_true_weighted), axis=-1))

    # Euclidean Distance from [0,0] to [177.8, 100] = 203.992
    norm_scale = ops.array([203.992], dtype="float32")

    # Normalizes loss values to the diagonal-- makes loss easier to interpret
    normalized_loss = ops.divide(loss, norm_scale) * 100

    return normalized_loss
