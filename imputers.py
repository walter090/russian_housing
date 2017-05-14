from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf


class KNNImputer(object):
    def __init__(self, k=5, missing_values='NaN'):
        self.missing_values = missing_values
        self.k = k

    def impute(self, data, inplace=False):
        """
        imputer function for knn imputer. find all data entries
        with missing data. go through each entry and apply knn regression
        :param inplace: if set to True, apply the changes to the passed 
        argument
        :param data: the full data set with missing data, the data must be
        preprocessed, i.e. does not contain any string/boolean values
        :return: a copy of the transformed data
        """
        """
        TODO add argument feature_name to enable impute one specific feature
        and enable the option between discrete and continuous values
        """
        nan_data = data[data.isnull().any(axis=1)]
        purged_data = data.dropna()  # purged_data + nan_data = data
        imp = KNeighborsRegressor(n_neighbors=self.k)
        new_data = nan_data.copy()

        for _, row in nan_data.iterrows():
            nan_cols = row[row.isnull()].index
            knn_train = purged_data.drop(nan_cols, axis=1)
            no_nan_row = row.drop(nan_cols)
            for target in nan_cols:
                imp.fit(knn_train, purged_data[target])
                if inplace:
                    data.loc[data['id'] == row['id']][target] = imp.predict(no_nan_row)
                else:
                    new_data.loc[data['id'] == row['id']][target] = imp.predict(no_nan_row)
        return purged_data.append(new_data)


class RegressionImputer(BaseEstimator, TransformerMixin):
    """
    another approach to missing data imputation: linear regression
    """

    def __init__(self, missing_values='NaN'):
        self.missing_values = missing_values

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError

    @staticmethod
    def input(data):
        """
        input function for the tensorflow regression model
        :param data: pandas data frame
        :return: a tuple of features and labels
        """
        continuous_features = []
        categorical_features = []

        for column in data.columns:
            if column.dtype == 'object':
                categorical_features.append(column)
            else:
                continuous_features.append(column)

        continuous_data = [{feature: tf.constant(data.values)} for feature in continuous_features]
        categorical_data = [{feature: tf.one_hot()
                             for feature in categorical_features}]
