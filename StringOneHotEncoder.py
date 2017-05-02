from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class StringOneHotEncoder(object):
    label_encoder_ = LabelEncoder()
    one_hot_encoder_ = OneHotEncoder()
    int_labels_ = []
    one_hot_ = []

    def fit(self, x):
        self.int_labels_ = self.label_encoder_.fit_transform(x)
        self.one_hot_ = self.one_hot_encoder_.fit_transform(self.int_labels_.reshape(-1, 1))
        return self

    def transform(self, x):
        return self.one_hot_encoder_.transform(self.label_encoder_.transform(x).reshape(-1, 1))

    def fit_transform(self, x):
        return self.fit(x).transform(x)
