"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.special import comb
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def kmeans(X, num_clusters):
    # Randomly choose clusters
    rng = np.random.default_rng()
    i = rng.choice(X.shape[0], size=num_clusters, replace=False)
    centers = X[i]

    while True:
        # Assign labels based on closest center
        labels = np.argmin(cdist(X, centers), axis=1)
        # Find new centers from means of points
        new_centers = np.array([X[labels == j].mean(axis=0) for j in range(num_clusters)])
        # Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return labels, centers


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


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']

    # Compute the Gaussian (RBF) similarity matrix
    pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
    affinity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    # Construct the graph Laplacian matrix
    L = laplacian(affinity_matrix, normed=True)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L, subset_by_index=[0, k-1])

    # Run K-means on the first k eigenvectors
    computed_labels, _ = kmeans(eigenvectors[:, :k], k)

    # Calculate SSE
    SSE = np.sum((data - np.mean(data, axis=0)) ** 2)
    # Calculate ARI
    ARI = adjusted_rand_score(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    data = np.load('question1_cluster_data.npy')
    labels = np.load('question1_cluster_labels.npy')


    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    sigma_values = np.linspace(0.1, 10, 5)
    groups = {}
    eigenvalues_group = []
    for i, sigma in enumerate(sigma_values):
        params_dict = {'sigma': sigma, 'k': 5}
        computed_labels, SSE, ARI, eigenvalues = spectral(data[1000*i:1000*(i+1)], labels[1000*i:1000*(i+1)],params_dict)
        groups[i] = {"sigma": sigma, "ARI": ARI, "SSE": SSE}
        eigenvalues_group.append(eigenvalues)

        if i == 0:  # Save the SSE of the first group for later access
            answers["1st group, SSE"] = SSE

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    #answers["1st group, SSE"] = {}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    sigmas = [group["sigma"] for group in groups.values()]
    ARI_values = [group["ARI"] for group in groups.values()]
    SSE_values = [group["SSE"] for group in groups.values()]
    eigenvalues = list(eigenvalues_group)

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
    plt.title(f'Scatter Plot with Largest ARI (σ = {groups[largest_ARI_index]["sigma"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with largest ARI"] = scatter_ARI

    plt.figure()
    scatter_SSE = plt.scatter(data_smallest_SSE[:, 0], data_smallest_SSE[:, 1], c=labels_smallest_SSE, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatter Plot with Smallest SSE (σ = {groups[smallest_SSE_index]["sigma"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with smallest SSE"] = scatter_SSE

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    sorted_eigenvalues = np.sort(eigenvalues)
    plt.figure()
    plot_eig = plt.plot(sorted_eigenvalues, marker='o')  # marker='o' makes the individual eigenvalues more visible
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues from Smallest to Largest')
    plt.grid(True)
    plt.show()
    answers["eigenvalue plot"] = plot_eig

    data_set_0 = [data[1000*0:1000*(0+1)], labels[1000*0:1000*(0+1)]]
    data_set_1 = [data[1000*0:1000*(1+1)], labels[1000*0:1000*(1+1)]]
    data_set_2 = [data[1000*0:1000*(2+1)], labels[1000*0:1000*(2+1)]]
    data_set_3 = [data[1000*0:1000*(3+1)], labels[1000*0:1000*(3+1)]]
    data_set_4 = [data[1000*0:1000*(4+1)], labels[1000*0:1000*(4+1)]]

    data_sets = [data_set_0, data_set_1, data_set_2, data_set_3, data_set_4]
    largest_ARI_parameters = {'sigma': sigma_values[largest_ARI_index], 'k': 5}
    ARIs = []
    SSEs = []
    for data, labels in data_sets:
        computed_labels, SSE, ARI, eigenvalues = spectral(data, labels, largest_ARI_parameters)
        ARIs.append(ARI)
        SSEs.append(SSE)


    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
