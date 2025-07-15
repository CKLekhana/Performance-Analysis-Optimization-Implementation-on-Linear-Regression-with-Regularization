"""
This python file defines all the data loaders and other utility functions required for data loading and processing. 
"""


# import libraries required
import random
import os
import numpy as np
import pandas as pd
from typing import Tuple
import opendatasets as od

# computing cluster size of the correlated subset of features 
def correlation_cluster_generator(n_features,  informative_indices) -> Tuple[np.ndarray]:
    """
    Finds the number of correlated clusters and the indexes in the clusters
    lets say there are 50 
    Args:
        n_features (int) : number of features in the dataset
        informative_indices (np.ndarray) : list of informative indices that the n_remaining features depend on

    Returns:
        correlated_cluster_idx [np.ndarray]: m x n matrix, correlated_cluster_idx[m][n] is the feature index belonging to cluster m.
    """
    
    try:
        n_informative = len(informative_indices)
        non_informative_indices = np.setdiff1d(np.arange(n_features), informative_indices)
        n_clusters = min(n_informative, len(non_informative_indices)//n_informative)
        
        # each cluster contains atleast one informative idx
        anchor_idx = np.random.choice(informative_indices, n_clusters, replace=False)
        
        # add each anchor_idx into correlation_index
        corr_cluster_idx = []
        
        for idx in anchor_idx:
            corr_cluster_idx.append([idx])
        
        # distribute non informative index in each of the clusters
        for i, idx in enumerate(non_informative_indices):
            cluster_idx = i % n_clusters
            corr_cluster_idx[cluster_idx].append(idx)
            
        clusters = [np.array(sorted(c)) for c in corr_cluster_idx]
        
        print("Correlation Clusters Generation Successfully") 
    except Exception as e:
        print("Correlation Clusters Generation Failed")
        print("Error: ", e)
        
    return clusters

def apply_cluster_relation(X, cluster_idx, informative_idx, multicollinearity_strength):
    """
    This function establishes the relation between the informative feature with other features in the correlation cluster

    Args:
        X (np.ndarray): raw input dataset created
        cluster_idx (np.ndarray): numpy array containing index in that particular cluster 
        informative_idx (int): informative feature in the cluster
        multicollinearity_strength (float): determines collinearity strength between features in the cluster
    """
    try:
        non_informative_idx = np.setdiff1d(cluster_idx, informative_idx)
        
        relations_dict = {
            1 : lambda x1, x2 : multicollinearity_strength*x1 + (1 - multicollinearity_strength)*x2 ,
            2 : lambda x1, x2 : multicollinearity_strength*x1 - (1 - multicollinearity_strength)*x2 , 
        }
            
        if len(non_informative_idx) in [0, 1]:
            return
            
        # values of the informative feature in the cluster
        anchor_values = X[:, informative_idx]
          
        for idx in non_informative_idx:
            X[:, idx] = relations_dict[np.random.randint(1,3)](X[:, idx], anchor_values)
            
        print("Apply correlation to clusters generated successfully!!")
    except (Exception) as e:
        print("Apply correlation to clusters generated failed due to: ")
        print("Error: ", e)
            

# Synthetic Data Generator
def generate_synthetic_dataset(n_samples: int = 100, 
                               n_features: int = 15, 
                               n_informative: int = 3, 
                               noise: float = 0.5, 
                               multicollinearity_strength: float = 0.5,
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame , np.ndarray]:
    """
    Generates synthetic dataset that aids in effective analyis of linear regression models 
    in a controlled structure. The dataset inherently contains high multicollinearity.
    
    Args:
        n_samples (int) : number of samples in the dataset
        n_features (int) : number of features in the dataset
        n_informative (int) : number of features that has true contribution
        noise (float) : introduces variability in the dataset
        multicollinearity_strength (float) : ranges between 0 and 1, indicating the correlation between the features 
        random_state (int) : seed for reproducibility

    Returns:
        Tuple[np.ndarray,np.ndarray , np.ndarray]: 
            - X (pd.DataFrame): input features to the model
            - y (pd.DataFrame): output variable (target)
            - true_beta (np.ndarray): true coefficients of the input features
    """
    
    print(f"Generating Synthetic Dataset with {n_samples} samples and {n_features} features.")
    np.random.seed(random_state)
    random.seed(random_state)
    
    try:
        # Fill n_samples with n_features values and stores as X, this is the input data to models
        X_raw = np.random.randn(n_samples, n_features)
        
        # informative indexes
        informative_indices = np.random.choice(np.arange(n_features), n_informative, replace=False)
        
        # Creating correlated feature subsets only if multicollinearity strength is greater than 1
        if multicollinearity_strength > 0:
            
            # correlation clusters
            correlation_cluster_indices = correlation_cluster_generator(n_features, informative_indices)
            
            # Designing correlations for each of the clusters
            for cluster_idxs in correlation_cluster_indices:
                informative_index = np.intersect1d(informative_indices, cluster_idxs)[0]
                apply_cluster_relation(X_raw, cluster_idxs, informative_index, multicollinearity_strength)
                        
        
        # Create coefficients for n_informative features
        true_beta = np.zeros(n_features)
        
        # all the synthetic dataset will have same distributions of coefficients
        true_beta[informative_indices] = np.random.uniform(-5, 5, n_informative)
        
        # adding the intercept coefficient to true_beta
        true_beta = np.insert(true_beta, 0, np.random.uniform(-5, 5))
        
        # since x0 = 1, add a column of ones in the beginning
        X_intercept = np.hstack((np.ones((n_samples,1)), X_raw))
        
        # Determine y using the coefficients and X determined so far
        y = X_intercept @ true_beta + noise * np.random.randn(n_samples)
        
        print("Synthetic Data Generation Successful")
        
        cols = [f"Feature {i+1}" for i in  range(n_features)]
        cols.insert(0, "Intercept")
        #print(cols)
        
        X_df = pd.DataFrame(X_intercept, columns=cols)
        y_df = pd.DataFrame(y, columns=["target"])
        
        return X_df, y_df, true_beta
    except (Exception) as e:
        
        print("Synthetic Data Generation Failed")
        print("Error: ", e)
        
    return None, None, None

# This function is responsible for generating or loading synthetic or real world datsets using the functions defined above
def data_loading_preprocessing(n_samples: int = 100, # required for synthetic data generation
                               n_features: int = 15, # required for synthetic data generation
                               n_informative: int = 3, # required for synthetic data generation
                               noise: float = 0.5, # required for synthetic data generation
                               multicollinearity_strength: float = 0.5, # required for synthetic data generation
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    
    try:
        
        X, y, true_beta = generate_synthetic_dataset(n_samples, n_features, n_informative, noise, multicollinearity_strength)
        return X, y, true_beta
        
    except Exception as e:
        print(f"Error in Loading & Processing Datasets. Error {e}")