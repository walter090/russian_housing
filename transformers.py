from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn_pandas import DataFrameMapper


class StringOneHotEncoder(TransformerMixin, BaseEstimator):
    """
    one hot encoder for string values
    inheritor of TransformerMixin, can be used in a sklearn pipeline
    """

    def __init__(self):
        """
        constructor
        """
        self.label_encoder_ = LabelEncoder()
        self.one_hot_encoder_ = OneHotEncoder()
        self.int_labels_ = []
        self.one_hot_ = []

    def fit(self, x):
        self.int_labels_ = self.label_encoder_.fit_transform(x)
        self.one_hot_ = self.one_hot_encoder_.fit_transform(self.int_labels_.reshape(-1, 1))
        return self

    def transform(self, x):
        return self.one_hot_encoder_.transform(self.label_encoder_.transform(x).reshape(-1, 1))

    def fit_transform(self, x, **kwargs):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        raise NotImplementedError


class LogNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, x, y=None, **fit_params):
        import numpy as np
        return np.log(x)


@DeprecationWarning
class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    extracts feature data for sklearn feature union
    
    (with the use of sklearn-pandas, this class does not seem to be
    necessary anymore)
    """

    def __init__(self, features):
        """
        constructor
        :param features: string value, names of the feature to extract
        """
        self.features = features

    def fit(self):
        return self

    def transform(self, x):
        return x[self.features]

    def fit_transform(self, x, y=None, **fit_params):
        return self.transform(x)


class FeatureMapper(BaseEstimator, TransformerMixin):
    def __int__(self, transformations):
        """
        constructor
        :param transformations: array. a list of pairs of string value feature name
        and transformations to be performed on them, this parameter will be fed into
        the DataFrameMapper
        :return: 
        """
        self.transformations = transformations

    def fit_transform(self, **kwargs):
        return DataFrameMapper(self.transformations)
