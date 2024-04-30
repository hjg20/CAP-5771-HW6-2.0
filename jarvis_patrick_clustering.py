"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from scipy.spatial.distance import cdist
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def adjusted_rand_score(labels_true, labels_pred):
    # Find the contingency table
    contingency_matrix = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true).size, np.unique(labels_pred).size))[0]

    # Sum the combinatorics for each row and column
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency_matrix, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency_matrix, axis=0))

    # Sum the combinatorics for the whole matrix
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency_matrix.flatten())

    # Calculate the expected index (as if the agreement is purely random)
    expected_index = sum_comb_c * sum_comb_k / comb(contingency_matrix.sum(), 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32] | None, float | None, float | None]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    
    k = int(params_dict['k'])
    smin = int(params_dict['smin'])
    
    # Step 1: Compute all pair-wise distances
    distances = cdist(data, data, metric='euclidean')
    
    # Step 2: Find k-nearest neighbors (excluding self)
    neighbors = np.argsort(distances, axis=1)[:, 1:k+1]
    
    # Step 3: Form shared neighbor connections based on 'smin'
    shared_neighbors = np.zeros_like(distances)
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            # Count the shared neighbors between points i and j
            shared_count = np.intersect1d(neighbors[i], neighbors[j]).shape[0]
            # Mark the shared neighbors if count exceeds smin
            if shared_count >= smin:
                shared_neighbors[i, j] = shared_neighbors[j, i] = 1
    
    # Step 4: Cluster points using a DBSCAN-like method based on the shared neighbors
    unvisited = set(range(len(data)))
    clusters = []
    while unvisited:
        # Randomly pick an unvisited point
        point = unvisited.pop()
        cluster = [point]
        points_to_visit = set(np.where(shared_neighbors[point] == 1)[0])
        while points_to_visit:
            point = points_to_visit.pop()
            if point in unvisited:
                unvisited.remove(point)
                cluster.append(point)
                new_neighbors = set(np.where(shared_neighbors[point] == 1)[0])
                points_to_visit |= new_neighbors
        clusters.append(cluster)
    
    # Step 5: Assign cluster labels
    computed_labels = np.zeros(len(data), dtype=int)
    for idx, cluster in enumerate(clusters):
        computed_labels[cluster] = idx
    
    # Step 6: Calculate SSE (optional)
    SSE = 0
    for cluster in clusters:
        points = data[cluster]
        centroid = np.mean(points, axis=0)
        SSE += np.sum((points - centroid) ** 2)
    
    # Step 7: Calculate ARI
    ARI = adjusted_rand_score(labels, computed_labels)
    
    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    k_values = np.linspace(3, 8, 5)
    smin_values = np.linspace(4, 10, 5)
    groups = {}
    for i in range(5):
        params_dict = {'k': k_values[i], 'smin': smin_values[i]}
        computed_labels, SSE, ARI = jarvis_patrick(data[1000*i:1000*(i+1)], labels[1000*i:1000*(i+1)],params_dict)
        groups[i] = {"k": params_dict['k'], "smin": params_dict['smin'], "ARI": ARI, "SSE": SSE}
        if i == 0:
            answers["1st group, SSE"] = SSE

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    # answers["1st group, SSE"] = {}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    largest_ARI_index = max(groups, key=lambda i: groups[i]["ARI"])
    data_largest_ARI = data[1000*largest_ARI_index:1000*(largest_ARI_index+1)]
    labels_largest_ARI = labels[1000*largest_ARI_index:1000*(largest_ARI_index+1)]

    smallest_SSE_index = min(groups, key=lambda i: groups[i]["SSE"])
    data_smallest_SSE = data[1000*smallest_SSE_index:1000*(smallest_SSE_index+1)]
    labels_smallest_SSE = labels[1000*smallest_SSE_index:1000*(smallest_SSE_index+1)]

    # Plot is the return value of a call to plt.scatter()
    plt.figure()
    scatter_ARI = plt.scatter(data_largest_ARI[:, 0], data_largest_ARI[:, 1], c=labels_largest_ARI, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatter Plot with Largest ARI (k = {groups[largest_ARI_index]["k"]}, smin = {groups[largest_ARI_index]["smin"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with largest ARI"] = scatter_ARI
    plt.savefig('Jarvis_ARI.png')
    plt.close()

    plt.figure()
    scatter_SSE = plt.scatter(data_smallest_SSE[:, 0], data_smallest_SSE[:, 1], c=labels_smallest_SSE, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatter Plot with Smallest SSE (k = {groups[smallest_SSE_index]["k"]}, smin = {groups[smallest_SSE_index]["smin"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with smallest SSE"] = scatter_SSE
    plt.savefig('Jarvis_SSE.png')
    plt.close()

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    for i in range(5):
        data_set = [data[1000*i:1000*(i+1)], labels[1000*i:1000*(i+1)]]
        largest_ARI_parameters = {'smin': smin_values[largest_ARI_index], 'k': k_values[largest_ARI_index]}
        ARIs = []
        SSEs = []
        _, SSE, ARI = jarvis_patrick(data_set[0], data_set[1], largest_ARI_parameters)
        ARIs.append(ARI)
        SSEs.append(SSE)

    # A single float
    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
