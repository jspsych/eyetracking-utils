import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random


"""Plots performance of model and displays average distance and variance of predictions.
 Args:
    dataset_size::[int]
        Size of dataset.
    test_data_size::[int]
        Size of test data.
    num_points::[int]
        Desired number of points to be plotted.
    dataset
      Dataset not shuffled or sliced.
    predictions
        Output of model based on test data.
"""
def plot_model_performance(dataset_size, test_data_size, num_points, dataset, predictions):

  test_data = list(dataset.skip(dataset_size - test_data_size))

  plt.xlim(0,100)
  plt.ylim(100,0)
  plt.rcParams.update({'font.size': 7})
  
  indexes = []
  for i in range(0, num_points):
    indexes += [random.randint(0, test_data_size)]
  
  distances = []
  for i in range(0, num_points):
    label = np.array(test_data[indexes[i]][1])
    lx = label[0]
    ly = label[1]

    px = predictions[:,0][indexes[i]]
    py = predictions[:,1][indexes[i]]

    plt.scatter(lx, ly, c="blue", zorder=2)
    plt.scatter(px, py, c="red", zorder=3)

    x_values = [lx, px]
    y_values = [ly, py]

    plt.plot(x_values, y_values, linestyle="--", zorder=1, color=(0.8, 0.8, 0.8))

    distance = np.array(normalized_weighted_euc_dist_loss([lx, ly], [px, py]))
    distances += [distance]

    plt.text(lx+2.0, ly+1.0, "D: {:0.2f}".format(distance[0]))
  
  distances_all = []
  predicted_points_x = []
  predicted_points_y = []
  for i in range(0, test_data_size):
    label = np.array(test_data[i][1])

    px = predictions[:,0][i]
    py = predictions[:,1][i]

    predicted_points_x += [px]
    predicted_points_y += [py]

    distance = np.array(normalized_weighted_euc_dist_loss(label, [px, py]))
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
