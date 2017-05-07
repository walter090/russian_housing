from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    extracts feature data for sklearn feature union
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


class FeatureDrop(BaseEstimator, TransformerMixin):
    def __init__(self):
        raise NotImplementedError
