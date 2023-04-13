from sklearn.cluster import AffinityPropagation
import numpy as np
from sklearn import metrics

"""
Major Failure:

This method cannot be applied with our data.
Our data point ~ 1.5M
RAM required for 1.5M X 1.5M array ~ 7TB 
"""

class AffinityClusterer:
    def __init__(self, TIMES, argument_dict):
        self.damping = argument_dict["damping"]
        self.max_iter = TIMES
        self.convergence_iter = argument_dict["convergence_iter"]
        self.affinity = AffinityPropagation(damping=self.damping, max_iter=self.max_iter, convergence_iter=self.convergence_iter, )

        self.centers = None
        self.indices = None

    def fit(self, x):  # x: (num_sample, num_feat)
        self.affinity = self.affinity.fit(x)

        print(
            "Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(x, self.affinity.predict(x), metric="sqeuclidean")
        )

        self.centers = self.affinity.cluster_centers_
        self.indices = self.affinity.cluster_centers_indices_

        # return np.array(score_container)[indices]

    def get_assignment(self, x):
        """
        x: (num_samples, num_feat)
        Returns: {n_samples,} Predict the closest cluster each sample in X belongs to. 
        """
        if self.centers and len (self.centers)>0:
            labelA = self.affinity.predict(x)
        else: 
            labelA = self.affinity.fit_predict(x)

            self.centers = self.affinity.cluster_centers_
            self.indices = self.affinity.cluster_centers_indices_
        return labelA

    def get_centroids(self, ):
        return self.centers


    """
    Method below is to compare with labels from other algorithms

    credits: https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py
    
    labelA, labelK: labels from different the algorithms
    """
    def compare_other_labels(self, labelA, labelK):
        n_clusters_ = len(self.centers)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labelA, labelK))
        print("Completeness: %0.3f" % metrics.completeness_score(labelA, labelK))
        print("V-measure: %0.3f" % metrics.v_measure_score(labelA, labelK))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labelA, labelK))
        print(
            "Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labelA, labelK)
        )