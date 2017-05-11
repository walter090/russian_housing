def get_missing(data, minimal=0.1, full=False):
    """
    return a list of features with missing data the ratio of whose number of 
    missing data entries over total data entries is larger than minimal
    :param data: pandas data frame
    :param minimal: set the minimal proportion of missing data to report
    :param full: if set to True, ignore minial, report all features with
    missing data
    :return: 
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
    """
    arguments same as get_missing, except for inplace
    :param data: 
    :param minimal: 
    :param full: 
    :param inplace: if set to True, apply the changes to the original data frame,
    otherwise return a new object
    :return: 
    """
    features_missing = [feature[0] for feature in get_missing(data, minimal, full)]
    if not inplace:
        return data.drop(features_missing, axis=1)
    else:
        data.drop(features_missing, axis=1, inplace=True)
        return None
