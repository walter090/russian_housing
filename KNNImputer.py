import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNImputer(object):
    def __init__(self, k=5, missing_values='NaN'):
        self.missing_values = missing_values

    def impute(self, data, inplace=False):
        """
        imputer function for knn imputer. find all data entries
        with missing data. group together entries with same features 
        missing; go through each group and apply knn regression
        :param inplace: if set to True, apply the changes to the passed 
        argument
        :param data: the full data set with missing data
        :return: a copy of the transformed data
        """
        nan_data = data[data.isnull().any(axis=1)]
        purged_data = data.dropna()

