import numpy as np


def onehot_to_seq(onehot):
    """
    onehot:
        (L,4)
        (1,L,4)
        (B,4,L)
        (B,L,4)
    return:
        list[str]  or  str
    """

    # ---------- single ----------
    if onehot.ndim == 2:
        # (L,4)
        idx_to_nuc = np.array(["A","C","G","T"])
        return "".join(idx_to_nuc[np.argmax(onehot, axis=1)])

    # ---------- batch ----------
    if onehot.ndim == 3:

        # (B,4,L) -> (B,L,4)
        if onehot.shape[1] == 4:
            onehot = np.transpose(onehot, (0,2,1))

        seqs = []
        idx_to_nuc = np.array(["A","C","G","T"])

        for s in onehot:
            seqs.append("".join(idx_to_nuc[np.argmax(s, axis=1)]))

        return seqs

# cited regLM
def get_percentiles(values, n_bins=None, qlist=None):
    """
    Return list of tokens for sequences by binning their associated values

    Args:
        values (list): Values for which to calculate percentiles
        n_bins (int): Number of equal bins into which to split values
        qlist (list): Quantiles to split values into

    Returns:
        List containing percentiles at which to split the values
    """
    # If given a number of bins, split the given values into equal bins.
    if n_bins is not None:
        assert n_bins < len(values)
        binwidth = 100 / n_bins
        qlist = np.arange(binwidth, 100, binwidth)

    # Find values that split the values by percentiles
    return np.percentile(values, qlist)


def get_label_tokens(values, percentiles):
    """
    Return labels for sequences given cutoff percentiles

    Args:
        values (list): Values for which to calculate percentiles
        percentiles (list): Percentiles at which to split values

    Returns:
        list containing label token corresponding to each value
    """
    return [str(x) for x in np.digitize(values, percentiles)]


def tokenize(df, cols, names, n_bins=None, qlist=None, percentiles=None):
    """
    Create labels for sequences by dividing them into bins

    Args:
        df (pd.DataFrame): Dataframe containing label values
        cols (list): Names of columns to tokenize
        names (list): Names to use for the returned tokens
        n_bins (int): Number of equal bins into which to split values
        qlist (list): Quantiles to split values into
        percentiles (dict): Dictionary containing columns from cols
            as keys, and lists of percentile values.

    Returns:
        df (pd.DataFrame): Original dataframe with additional columns containing
        tokenized labels
    """
    # Get percentiles
    if percentiles is None:
        percentiles = dict()
        for col in cols:
            percentiles[col] = get_percentiles(df[col], n_bins=n_bins, qlist=qlist)
            print(col, percentiles[col].tolist())

    # Add a column to contain the label
    df["label"] = [""] * len(df)

    # Fill in tokens and labels
    for name, col in zip(names, cols):
        df[name + "_token"] = get_label_tokens(df[col], percentiles[col])
        df["label"] = df["label"] + df[name + "_token"]

    return df