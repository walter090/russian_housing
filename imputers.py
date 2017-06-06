from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class Imputer(object):
    def __init__(self, strategy='regression'):
        """Constructor for imputer class

        Args:
            strategy: Imputation strategy, available strategies:
                'regression', 'knn', 'mean'. Default strategy is 'knn'
        """
        self.strategy = strategy

    def impute(self, data, exclude=None, categorical=None, inplace=False, k=10):
        """This function initiates the imputation

        The function takes a Pandas data frame, and replace the values
        specified in the constructor with imputed values

        Args:
            data (pandas.DataFrame): DataFrame, the data to be transformed
            exclude (List[str]): list of strings, names of columns whose missing values will
                not be imputed, default None
            categorical (List[str]): list of strings, names of columns where the values are
                categorical instead of continues, these columns will not be used in training,
                default None
            inplace (bool): Boolean, if set to True, the original data frame will be imputed,
                default False
            k (int): Argument for when the imputation strategy is knn, this argument set the k
                parameter in KNN algorithm, default 10. If the strategy is other that knn, this
                argument is ignored
        Returns:
            None or new data frame new_df, depends on if inplace argument is set to True or False
        """
        nan_data = data[data.isnull().any(axis=1)]  # data frame columns that has missing values
        purged_data = data.dropna(axis=1, inplace=False).drop(categorical, inplace=False)
        # purged_data + nan_data = data, i.e. intact data
        # purged_data will serve as a set of "training features" in the imputation process
        purged_train = purged_data[nan_data.notnull().any(axis=0)]
        nan_train = nan_data[nan_data.notnull().any(axis=0)]
        # purged_train and nan_train are part of the nan and purged data that does not have
        # missing data, therefore used for training the imputer
        purged_target = purged_data[nan_data.isnull().any(axis=0)]

        cols_with_missing_data = list(nan_data)
        rows_missing_by_col = {}
        data_missing = data.isnull()
        for col in cols_with_missing_data:
            rows_missing_by_col[col] = data[data_missing[col] == True].index.tolist()
        # rows_missing_by_col is now a python dictionary that contain information about
        # rows with missing data in respect to each column name
        # {'column_name': [list of row indices]}
        # this will be useful when replacing missing data with imputed data

        if self.strategy == 'knn':
            clf = KNeighborsClassifier(n_neighbors=k)
            rgr = KNeighborsRegressor(n_neighbors=k)
        else:
            raise NotImplementedError

        new_df = data.copy()

        for col_name in nan_data:
            if col_name in exclude:
                continue
            if col_name in categorical:
                clf.fit(purged_train, nan_train[col_name])
                pred = clf.predict(purged_target)
            else:
                rgr.fit(purged_train, nan_train[col_name])
                pred = rgr.predict(purged_target)

            if inplace:
                data.loc[rows_missing_by_col[col_name], col_name] = pred
            else:
                new_df.loc[rows_missing_by_col[col_name], col_name] = pred

        if not inplace:
            return new_df
