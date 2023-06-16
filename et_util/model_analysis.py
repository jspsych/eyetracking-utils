import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random

def normalized_weighted_euc_dist(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Weighting treats y-axis as unit-scale and creates a rectangle that's 177.8x100 units.  
    x_weight = tf.constant([1.778, 1.0], dtype=tf.float32)

    # Multiply x-coordinate by 16/9 = 1.778
    y_true_weighted = tf.math.multiply(x_weight, y_true)
    y_pred_weighted = tf.math.multiply(x_weight, y_pred)

    # Calculate Euclidean distance with weighted coordinates
    loss = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred_weighted - y_true_weighted), axis=-1))

    # Euclidean Distance from [0,0] to [177.8, 100] = 203.992
    norm_scale = tf.constant([203.992], dtype=tf.float32)

    # Normalizes loss values to the diagonal-- makes loss easier to interpret
    normalized_loss = tf.math.divide(loss, norm_scale) * 100

    return normalized_loss
    
def plot_model_performance(num_points, test_data, predictions, function):
   """
   Plots performance of model and displays average distance and variance of predictions.
    
   :param num_points: number of points to be plotted
   :param test_data: test data used to make predictions
   :param predictions: output of model based on test data
   :param function: function that outputs array of labels for data of certain shape
   """
   
   test_data_size = 0
   for element in test_data:
     test_data_size += 1
   test_labels_arr = function(test_data)
   
   plt.xlim(0,100)
   plt.ylim(100,0)
   plt.rcParams.update({'font.size': 7})
  
   indexes = []
   for i in range(0, num_points):
     indexes += [random.randint(0, test_data_size)]
  
   distances = []
   for i in range(0, num_points):
     label = np.array(test_labels_arr[indexes[i]])
     lx = label[0]
     ly = label[1]

     px = predictions[:,0][indexes[i]]
     py = predictions[:,1][indexes[i]]

     plt.scatter(lx, ly, c="blue", zorder=2)
     plt.scatter(px, py, c="red", zorder=3)

     x_values = [lx, px]
     y_values = [ly, py]

     plt.plot(x_values, y_values, linestyle="--", zorder=1, color=(0.8, 0.8, 0.8))

     distance = np.array(normalized_weighted_euc_dist([lx, ly], [px, py]))
     distances += [distance]

     plt.text(lx+2.0, ly+1.0, "D: {:0.2f}".format(distance[0]))
  
   distances_all = []
   predicted_points_x = []
   predicted_points_y = []
   for i in range(0, test_data_size):
     label = np.array(test_labels_arr[i])

     px = predictions[:,0][i]
     py = predictions[:,1][i]

     predicted_points_x += [px]
     predicted_points_y += [py]

     distance = np.array(normalized_weighted_euc_dist(label, [px, py]))
     distances_all += [distance]

   avg_distance = np.mean(distances_all)
   avg_x = np.mean(predicted_points_x)
   avg_y = np.mean(predicted_points_y)

   square_diff_x = []
   square_diff_y = []
   for i in range(0, test_data_size):
     square_diff_x += [np.square(predicted_points_x[i] - avg_x)]
     square_diff_y += [np.square(predicted_points_y[i] - avg_y)]
  
   variance_x = sum(square_diff_x) / test_data_size
   variance_y = sum(square_diff_y) / test_data_size

   plt.text(80, 16, "Avg. Dist: {:0.2f}".format(avg_distance), color = "green", weight="bold")
   plt.text(75, 21, "Var. of Pred x: {:0.2f}".format(variance_x), color = "green", weight="bold")
   plt.text(75, 26, "Var. of Pred y: {:0.2f}".format(variance_y), color = "green", weight="bold")
   blue_patch = mpatches.Patch(color="blue", label="Labels")
   red_patch = mpatches.Patch(color='red', label='Predictions')
   plt.legend(loc="upper right", handles=[red_patch, blue_patch])
   plt.show()


def gen_test_arr_landmarks(test_data):
  """Helper function for plot_model_performance that generates
  array of test points from test data in format landmarks, label
  :param test_data:
  :return: array of test points"""
  arr = [label for landmarks, label in test_data]
  return arr
