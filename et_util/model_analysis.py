import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from et_util.custom_loss import normalized_weighted_euc_dist
    
def plot_model_performance(num_points, test_data, predictions):
  """
  Plots performance of model and displays average distance and variance of predictions.

  :param num_points: number of points to be plotted
  :param test_data: test data used to make predictions
  :param predictions: output of model based on test data
  """

  test_data_size = len(predictions)
  test_labels_arr = [elm[-2] for elm in test_data]

  plt.xlim(0,100)
  plt.ylim(100,0)
  plt.rcParams.update({'font.size': 7})

  indexes = []
  for i in range(0, num_points):
    indexes.append(random.randint(0, test_data_size - 1))

  distances = []
  for i in range(0, num_points):
    label = np.array(test_labels_arr[indexes[i]])
    lx = label[0]
    ly = label[1]

    px = predictions[indexes[i]][0][0]
    py = predictions[indexes[i]][0][1]

    plt.scatter(lx, ly, edgecolors="black", facecolors="none", zorder=2)
    plt.scatter(px, py, c="red", zorder=3)

    x_values = [lx, px]
    y_values = [ly, py]

    plt.plot(x_values, y_values, linestyle="--", zorder=1, color=(0.8, 0.8, 0.8))

    distance = np.array(normalized_weighted_euc_dist([lx, ly], [px, py]))
    distances += [distance]

    plt.text(lx-2.5, ly+4.0, "{:0.2f}".format(distance[0]), fontsize=6)

  distances_all = []
  predicted_points_x = []
  predicted_points_y = []
  for i in range(0, test_data_size):
    label = np.array(test_labels_arr[i])

    px = predictions[i][0][0]
    py = predictions[i][0][1]

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

  plt.text(80, 16, "Avg. Dist: {:0.2f}".format(avg_distance), color = "black", weight="bold")
  plt.text(75, 21, "Var. of Pred x: {:0.2f}".format(variance_x), color = "black", weight="bold")
  plt.text(75, 26, "Var. of Pred y: {:0.2f}".format(variance_y), color = "black", weight="bold")
  black_patch = mpatches.Patch(color="black", label="Labels")
  red_patch = mpatches.Patch(color='red', label='Predictions')
  plt.legend(loc="upper right", handles=[red_patch, black_patch])
  plt.show()
