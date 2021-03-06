def get_missing(data, minimal=0.1, full=False):
    """return a list of features with missing data the ratio of whose number of
    missing data entries over total data entries is larger than minimal.

    Args:
        data: Pandas data frame
        minimal: Set the minimal proportion of missing data to report
        full: If set to True, ignore minial, report all features with
            missing data
    Returns:
        List of features with missing data, sorted
    """
    n_rows = data.shape[0]
    null_sum = data.isnull().sum()
    null_tuple = zip(null_sum.index, null_sum)
    return sorted([null_feature for null_feature in null_tuple
                   if null_feature[1] > n_rows * minimal], key=lambda t: t[1], reverse=True) if not full \
        else sorted([null_feature for null_feature
                     in null_tuple if null_feature[1] > 0],
                    key=lambda t: t[1],
                    reverse=True)


def remove_missing(data, minimal=0.1, full=False, inplace=False):
    """Arguments same as get_missing, except for inplace

    Args:
        data: Data to have missing data removed
        minimal: Amount of data larger than minimal will be removed
        full: If set to True, ignore minial, report all features with
            missing data
        inplace: if set to True, apply the changes to the original data frame,
            otherwise return a new object
    Returns:
        None or modified original data
    """
    features_missing = [feature[0] for feature in get_missing(data, minimal, full)]
    if not inplace:
        return data.drop(features_missing, axis=1)
    else:
        data.drop(features_missing, axis=1, inplace=True)
        return None


def make_transformations(data, sub_tran, filler=None):
    """This function is written for the ease of use of sklearn-pandas.

    sklearn-pandas DataFrameMapper only keeps features that are in the 
    transformation lists, columns that does not need transformations need
    to specify transformation as None. this function also provides an option
    to fill these features with the same transformer, specified by filler
    argument, default is None, not transformations

    Args:
        data: pandas data frame to extract features from
        sub_tran: array. a list of pairs of feature names
            and transformations.
        filler: value or transformation to for filled features, default is None,
            filler argument takes a class
    Returns:
        complete:
    """
    sub_features = [trans_pair[0] for trans_pair in sub_tran]
    appending_features = [feature for feature in list(data) if feature not in sub_features]
    filled = [(feature, filler if filler is None else filler()) for feature in appending_features]
    complete = []
    complete.extend(sub_tran)
    complete.extend(filled)
    return complete
