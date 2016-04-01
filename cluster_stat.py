from __future__ import print_function, absolute_import, division
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import yaml
import argparse


"""
   Module to analyse the data to suggest the numbers of clusters and create the cluster
"""


def get_optimal_cluster_num(data, max_cluster_num=10, algorithm='kMeans', max_iter=300, distance_metric='euclidean'):
    """
    Get the optimal number of clusters for the given dataset -
    Its a research topic so there might be multiple method for this. Its a research problem because most of the standard
    research method does not work well in practice
     Standard Method -
    1. Within Sum of Square just talks about how close the member of clusters are to each other, it does not talk about
        a. how well clusters are separated
        b. cluster size
        c.it decreases wit increase in number of cluster, but the rate of decrease gets slower when number of
          clusters gets higer
    2. Silhoutte Score - this score talks about how well clusters are separated. A higher value is better but the value is
       always observed to be highest when number of cluster is 2, which is not ideal
    Other Method - These methods will be tried as part of this research
    1. Variance in the size of cluster
    2. Variance in the combiend Z-score
    3. A combination of variance of cluster, z-score, silhoutte score, within sum of square
    :param data: data on which we will perform clustering, it should be indexed and scaled if scaling option has been
                 chosen by the user
    :return:
    """
    # TODO - Analyse the result and come up with the method which can give the optimal cluster size
    # TODO - Try out the information criterion approach and information theoretic approach

    cluster_statistics = pd.DataFrame(columns=['Num_Cluster', 'Within_Sum_Of_Square', 'Avg_Sil_Score',
                                               'Sil_Score_Std', 'Z_Score_Std', 'Cluster_Size_Std'])
    tot_sum_of_squares = get_tot_sum_of_squares(data)
    for i in range(2, max_cluster_num+1):
        if algorithm == 'kMeans':
            cluster_stat = k_means_cluster_stat(data, i, tot_sum_of_squares, max_iter, distance_metric)
            cluster_statistics.loc[i-2] = cluster_stat
    return cluster_statistics


def k_means_cluster_stat(data, num_cluster, tot_sum_of_square, max_iter=300, distance_metric='euclidean'):
    """
    Returns the clustering statistics for kMeans clustering for given number of clusters
    :param data: data on which clustering will be done
    :param num_cluster: number of clusters to be created
    :param max_iter: number of iteration to be performed before kMeans stops, either by convergance,
                     or by hitting this limit
    :param distance_metric: which distance metric should be used for clustering
    :return: various clustering statistics from given data and number of cluster
    """
    # TODO - Discuss with Kamrul-  Why kMeans can't be done on Manhattan distance ?
    k_means = KMeans(n_clusters=num_cluster, init='k-means++', n_init=10, max_iter=max_iter, random_state=100)
    k_means.fit(data.values)
    within_sum_of_square_ratio = k_means.inertia_/tot_sum_of_square
    cluster_label = k_means.labels_
    sil_score = get_average_silhouette_score(data, cluster_label, distance_metric, 10, 1000)
    z_score_table = get_z_score(data, cluster_label)
    z_score_std = get_z_score_variance(z_score_table)
    cluster_size_std = get_cluster_size_std(cluster_label)
    return [num_cluster, within_sum_of_square_ratio, sil_score[0],sil_score[1], z_score_std, cluster_size_std]


def get_z_score(data, labels):
    """
    Calculate the z-score of various features for a given cluster
    z-score is calculated as (mean of cluster feature - overall feature mean)/ feature standard deviation
    :param data: Pandas Data frame data used for clustering
    :param labels: Array of cluster labels
    :return: Pandas Data frame which contains cluster z_score for all features used for clustering
    """
    data['label'] = labels
    tot_label = np.max(data['label']) + 1
    df_col_names = ['Features'] + ['z_score_cluster_' + str(i) for i in range(tot_label)]
    z_score_table = pd.DataFrame(columns=df_col_names)
    i = 0
    for df_col in data.columns:
        if df_col != 'label':
            z_score = []
            for l in range(tot_label):
                score = (np.mean(data[data.label == l][df_col]) - np.mean(data[df_col])/np.std(data[df_col]))
                z_score.append(score)
            z_score_table.loc[i] = [df_col] + z_score
            i += 1
    return z_score_table


def get_z_score_variance(z_score_table):
    """
    Returns the overall standard standard deviation (sd) in z-score. To calculate overall sd, we first calculate the
    sd for each feature across the cluster and then sum up the sd of each feature to get overall sd
    :param z_score_table: Data frame containing feature z-score for all clusters
    :return: overall standard deviation of z-score
    """
    z_score_table.set_index('Features', drop=True, inplace=True)
    var = np.std(z_score_table, axis=1)
    return sum(var)


def get_cluster_size_std(labels):
    """
    Calculate the standard deviation in the size of generated clusters
    :param labels: Array containing cluster label
    :return:
    """
    cluster_size = []
    for i in range(np.max(labels)+1):
        cluster_size.append(len(labels[labels == i])*1.0/len(labels))
    return np.std(cluster_size)


def get_average_silhouette_score(data, labels, distance_metric='euclidean', sample_number=10, sample_size=1000):
    """
    Calculate average silhouette score and variance from multiple samples
    :param data: Data Frame used for clustering
    :param labels: Array contains cluster labales
    :param distance_metric: distance metric to be used for calculating silhouette score, default 'euclidean'
    :param sample_number: int - number of samples to be created for calculating average silhoeutte score, default 10
    :param sample_size: int - size of each sample
    :return: average silhouette score and it's standard deviation
    """
    sil_score = []
    for i in range(sample_number+1):
        random_index = np.random.randint(0, data.shape[0], sample_size)
        sil_score.append(silhouette_score(data.iloc[random_index], labels[random_index], metric=distance_metric))
    return [np.average(sil_score), np.std(sil_score)]


def get_tot_sum_of_squares(data):
    """
    :param data: data to be used for clustering
    :return: total sum of squares
    """
    data_matrix = data.as_matrix(columns=None)
    mean_matrix = np.mean(data_matrix, axis=0)
    tot_sum_of_squares = 0
    for i in range(data.shape[1]):
        tot_sum_of_squares += (data_matrix[:, i] - mean_matrix[i]).dot((data_matrix[:, i] - mean_matrix[i]).T)
    return tot_sum_of_squares


def create_k_means_cluster(data_to_fit, data_to_predict, num_cluster, max_iter=300, distance_metric='euclidean'):
    """
    Returns the clustering statistics for kMeans clustering for given number of clusters
    :param data_to_fit: data on which clustering will be done
    :param num_cluster: number of clusters to be created
    :param max_iter: number of iteration to be performed before kMeans stops, either by convergance,
                     or by hitting this limit
    :param distance_metric: which distance metric should be used for clustering
    :return: various clustering statistics from given data and number of cluster
    """
    # TODO - Discuss with Kamrul-  Why kMeans can't be done on Manhattan distance ?
    k_means = KMeans(n_clusters=num_cluster, init='k-means++', n_init=10, max_iter=max_iter, random_state=100)
    k_means.fit(data_to_fit.values)
    data_to_fit['cluster_label'] = k_means.labels_
    data_to_fit['fit_label'] = "Fitted"
    if not data_to_predict.empty:
        data_to_predict['cluster_label'] = k_means.predict(data_to_predict)
        data_to_predict['fit_label'] = "Predicted"
    return [data_to_fit, data_to_predict]


def get_file_name(params, initial_name):
    if params['use_categorical_field']:
        return initial_name + params['feature_selection_method'] + '_cataegorical.csv'
    else:
        return initial_name + params['feature_selection_method'] + '.csv'


def get_cluster_stats_table(pre_scaled_data):
    tot_label = len(np.unique(pre_scaled_data.cluster_label))
    table_columns = ['feature'] + ['z_score_cluster_' + str(i) for i in range(tot_label)] + \
                    ['mean_cluster_' + str(i) for i in range(tot_label)] +\
                    ['min_cluster_' + str(i) for i in range(tot_label)] + \
                    ['max_cluster_' + str(i) for i in range(tot_label)]
    score_table = pd.DataFrame(columns=table_columns)
    i = 0
    for df_col in pre_scaled_data.columns:
        if df_col != 'label' and df_col != "CUST_CODE" and df_col != "fit_label":
            z_score = []
            mean_score = []
            min_score = []
            max_score = []
            for l in range(tot_label):
                z_score.append((np.mean(pre_scaled_data[pre_scaled_data.cluster_label == l][df_col]) -
                                np.mean(pre_scaled_data[df_col])/np.std(pre_scaled_data[df_col])))
                mean_score.append(np.mean(pre_scaled_data[pre_scaled_data.cluster_label == l][df_col]))
                min_score.append(np.min(pre_scaled_data[pre_scaled_data.cluster_label == l][df_col]))
                max_score.append(np.max(pre_scaled_data[pre_scaled_data.cluster_label == l][df_col]))
            score_table.loc[i] = [df_col] + z_score + mean_score + min_score + max_score
            i += 1
    return score_table


def get_combined_report(pre_scaled_data, combined_data):
    """
    generate the report which contains the z-score, mean, min, max value for each feature. This will be helpful
    in understanding the clustering result in much better manner
    :return:
    """
    # TODO Implement this method, after deciding about whether the report should be generated from original data or from
    # TODO the data used for segmentation i.e. scaled data
    drop_cols = [c for c in pre_scaled_data.columns if c not in list(combined_data.columns)]
    pre_scaled_data.drop(drop_cols, axis=1, inplace=True)
    combined_data = combined_data[['CUST_CODE', 'cluster_label']]
    pre_scaled_data = pd.merge(pre_scaled_data, combined_data, on="CUST_CODE")
    return get_cluster_stats_table(pre_scaled_data)


def get_individual_and_combined_report(pre_scaled_data, combined_data):
    """
    generate three reports -
    1. report for fitted data
    2. report for predicted data
    3. report for combined data
    :return:
    """
    drop_cols = [c for c in pre_scaled_data.columns if c not in list(combined_data.columns)]
    pre_scaled_data.drop(drop_cols, axis=1, inplace=True)
    combined_data = combined_data[['CUST_CODE', 'cluster_label', 'fit_label']]
    pre_scaled_data = pd.merge(pre_scaled_data, combined_data, on="CUST_CODE")
    return [get_cluster_stats_table(pre_scaled_data[pre_scaled_data.fit_label == "Fitted"]),
            get_cluster_stats_table(pre_scaled_data[pre_scaled_data.fit_label == "Predicted"]),
            get_cluster_stats_table(pre_scaled_data)]


