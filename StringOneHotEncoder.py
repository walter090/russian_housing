from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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
