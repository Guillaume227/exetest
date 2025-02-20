import numpy as np
import pandas as pd
import typing
import functools


def load_df(file_path, ignore_cols=None, filter_cols=None):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_feather(file_path)

    if filter_cols:
        df = df[filter_cols]

    if ignore_cols:
        return df.loc[:, ~df.columns.isin(ignore_cols)]
    else:
        return df


def is_close(a, b, **kwargs):
    return np.isclose(a=b, b=a, **kwargs)


def uncentered_correlation(df1, df2):
    """
    Compute the uncentered correlation (cosine similarity) between corresponding columns of two DataFrames.
    """
    assert df1.shape == df2.shape, "DataFrames must have the same shape"

    # Normalize the dataframes
    df1_norm = np.linalg.norm(df1, axis=0)
    df2_norm = np.linalg.norm(df2, axis=0)

    # Compute the dot product
    dot_product = np.sum(df1 * df2, axis=0)

    # Compute the uncentered correlation
    uncentered_corr = dot_product / (df1_norm * df2_norm)

    return pd.Series(uncentered_corr, index=df1.columns)


class DFComparator:

    def __init__(self,
                 ignore_cols=None,
                 filter_cols=None,
                 verbose: bool = True,
                 num_diffs: int = 10,
                 **np_close_kwargs):
        """
        :param ignore_cols: columns to ignore during comparison
        :param verbose:
        :param num_diffs: number of diffs to display
        :param np_close_kwargs: np.allclose() kwargs to specify tolerance
        """
        self.ignore_cols = ignore_cols or []
        self.filter_cols = filter_cols or []
        self.verbose = verbose
        self.np_close_kwargs = np_close_kwargs
        self.num_diffs_to_display = num_diffs

    def description(self) -> str:
        if self.ignore_cols:
            return f"ignoring columns: {self.ignore_cols}"
        else:
            return ''

    def __call__(self, df_path1, df_path2) -> bool:
        df1 = load_df(file_path=df_path1, ignore_cols=self.ignore_cols, filter_cols=self.filter_cols)
        df2 = load_df(file_path=df_path2, ignore_cols=self.ignore_cols, filter_cols=self.filter_cols)
        return self.compare_dataframes(df1, df2).all()

    def compare_dataframes(self,
                           df1,
                           df2,
                           include_cols: typing.List[str] = None,
                           exclude_cols: typing.List[str] = None,
                           show_context_cols: bool = True) -> np.array:
        """
        :return: mask of rows with differences (True if row is the same)
        """

        exclude_cols = exclude_cols or []

        if self.filter_cols:
            df1 = df1.loc[:, self.filter_cols]
            df2 = df2.loc[:, self.filter_cols]

        if df1.equals(df2):
            return np.array([True] * df1.shape[0])

        shape_differs = df1.shape != df2.shape
        if shape_differs and self.verbose:
            print('df1 shape:', df1.shape)
            print('df2 shape:', df2.shape)

        columns_differ = False
        df1_only_cols = set(df1.columns.difference(df2.columns).values)
        if include_cols is not None:
            df1_only_cols.intersection_update(include_cols)
        df1_only_cols.difference_update(exclude_cols)
        if df1_only_cols:
            columns_differ = True
            if self.verbose:
                print(len(df1_only_cols), 'cols only in df1:', df1_only_cols)

        df2_only_cols = set(df2.columns.difference(df1.columns).values)
        if include_cols is not None:
            df2_only_cols.intersection_update(include_cols)
        df2_only_cols.difference_update(exclude_cols)
        if df2_only_cols:
            columns_differ = True
            if self.verbose:
                print(len(df2_only_cols), 'cols only in df2:', df2_only_cols)
                print()

        if include_cols is None:
            include_cols = set(df1.columns.to_list()).union(df2.columns.to_list())

        num_row_difference = 0
        if shape_differs or columns_differ:
            common_cols = df1.columns.intersection(df2.columns).values
            if self.verbose:
                if len(common_cols) != len(df1.columns) or len(common_cols) != len(df2.columns):
                    print(len(common_cols), 'common cols:', common_cols)
                    print()

            if common_cols.any():
                df1 = df1.loc[:, common_cols]
                df2 = df2.loc[:, common_cols]
            else:
                # all rows differ, no common cols
                return np.array([False] * df1.shape[0])

            num_rows1 = df1.shape[0]
            num_rows2 = df2.shape[0]
            if num_rows1 != num_rows2:
                num_row_difference = abs(num_rows1 - num_rows2)
                print(f"number of rows differ: {num_rows1} vs {num_rows2}")
                if num_rows1 > num_rows2:
                    df1 = df1.head(num_rows2)
                elif num_rows2 > num_rows1:
                    df2 = df2.head(num_rows1)

        for df in df1, df2:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df1_nans = df1.isna()
        df2_nans = df2.isna()
        differing_nan_mask = df1_nans ^ df2_nans

        nan_col_mask = differing_nan_mask.any(axis=0)

        diff_mask = nan_col_mask

        if cols_with_nans_diffs := nan_col_mask[nan_col_mask].index.to_list():
            nan_row_mask = differing_nan_mask.any(axis=1)
            print(len(cols_with_nans_diffs), 'cols with nan differences:', ' '.join(cols_with_nans_diffs))
            if self.verbose and self.num_diffs_to_display:
                print_df_diff(df1 if show_context_cols else df1[cols_with_nans_diffs],
                              df2 if show_context_cols else df2[cols_with_nans_diffs],
                              diff_mask=nan_row_mask,
                              num_diffs_to_display=self.num_diffs_to_display,
                              message='nans',
                              print_largest=False)
        else:
            print('no cols with Nans difference')

        matching_nan_mask = (df1_nans & df2_nans).any(axis=1)
        num_matching_nans = matching_nan_mask.sum()
        if self.verbose and num_matching_nans:
            print("there are", num_matching_nans, "matching nans")

        cols_with_diffs = [] #cols_with_nans_diffs
        for col in df1.columns:
            if col not in include_cols or col in exclude_cols:
                continue

            if df1[col].dtype != 'category' and np.issubdtype(df1[col].dtype, np.number) and not np.issubdtype(df1[col].dtype, np.integer)\
                                            and np.issubdtype(df2[col].dtype, np.number) and not np.issubdtype(df2[col].dtype, np.integer):
                # use numerical comparison
                if not np.allclose(df1[col].values, df2[col].values, **self.np_close_kwargs):
                    cols_with_diffs.append(col)
            else:
                if not np.equal(df1[col].values, df2[col].values).all():
                    cols_with_diffs.append(col)

        if cols_with_diffs:
            numerical_diff_cols = []
            non_numerical_diff_cols = []
            for col in cols_with_diffs:
                if np.issubdtype(df1[col].dtype, np.number) and \
                   np.issubdtype(df2[col].dtype, np.number):
                    numerical_diff_cols.append(col)
                else:
                    non_numerical_diff_cols.append(col)

            if numerical_diff_cols and self.verbose:
                single_value_cols = []
                with pd.option_context("display.float_format", "{:.2f}".format):
                    print()
                    print(f'correlation of numerical cols ({df1.shape[0]} rows):')

                    correlation_cols = []
                    for col in numerical_diff_cols:
                        is_unique_1 = df1[col].nunique() == 1
                        is_unique_2 = df2[col].nunique() == 1
                        if is_unique_1 or is_unique_2:
                            single_value_cols.append(col)
                        else:
                            correlation_cols.append(col)

                    if single_value_cols:
                        print("Cannot compute corr for cols with constant values:", " ".join(single_value_cols))

                    num_diff_df1 = df1[correlation_cols]
                    num_diff_df2 = df2[correlation_cols]

                    uncentered_corrs = uncentered_correlation(num_diff_df1, num_diff_df2)
                    centered_corrs = num_diff_df1.corrwith(num_diff_df2)
                    # combine both correlations into a single Dataframe
                    corrs_df = pd.DataFrame({'Centered Corr': centered_corrs, 'Uncentered Corr': uncentered_corrs})

                    print(corrs_df[~corrs_df.isna()].sort_values(ascending=False, by='Centered Corr').to_string())
                    if corrs_df.shape[0] > 16:
                        print(f'(on {df1.shape[0]} rows)')
                    print()

                if numerical_diff_cols:
                    df1_with_diff = df1[numerical_diff_cols]
                    df2_with_diff = df2[numerical_diff_cols]
                    diff_mask_numerical = ~(df1_with_diff - df2_with_diff).apply(
                        functools.partial(is_close, b=0, **self.np_close_kwargs))
                    diff_mask_numerical = diff_mask_numerical.any(axis=1)
                    diff_mask |= diff_mask_numerical
                    if self.verbose:
                        with pd.option_context("display.float_format", "{:g}".format):
                            print_df_diff(df1 if show_context_cols else df1_with_diff,
                                          df2 if show_context_cols else df2_with_diff,
                                          diff_mask=diff_mask_numerical,
                                          num_diffs_to_display=self.num_diffs_to_display,
                                          message='numerical')

                if non_numerical_diff_cols:
                    df1_with_diff = df1[non_numerical_diff_cols]
                    df2_with_diff = df2[non_numerical_diff_cols]
                    diff_mask_non_numerical = (df1_with_diff != df2_with_diff).any(axis=1)
                    diff_mask |= diff_mask_non_numerical

                    if self.verbose and self.num_diffs_to_display:
                        print_df_diff(df1 if show_context_cols else df1_with_diff,
                                      df2 if show_context_cols else df2_with_diff,
                                      diff_mask=diff_mask_non_numerical,
                                      num_diffs_to_display=self.num_diffs_to_display,
                                      message='non-numerical')

        if num_row_difference:
            extra_false = np.full(num_row_difference, True, dtype=bool)
            diff_mask = np.concatenate((diff_mask, extra_false))
        elif hasattr(diff_mask, 'to_numpy'):
            diff_mask = diff_mask.to_numpy()

        return ~diff_mask


def print_df_diff(df1, df2, diff_mask,
                  num_diffs_to_display: int,
                  message: str,
                  atol: float = None,
                  print_largest: bool = True):


    if num_diffs_to_display > 0:
        msg = f'first'
        func_name = 'head'
    elif num_diffs_to_display < 0:
        msg = f'last'
        func_name = 'tail'
    else:
        return

    num_diffs = int(diff_mask.sum())
    num_diffs_to_display = min(df1.shape[0], abs(num_diffs_to_display), num_diffs)
    msg += f' {num_diffs_to_display} differing rows'

    masked_df1 = df1[diff_mask]
    masked_df2 = df2[diff_mask]

    diff_cols = []
    diff_df = pd.DataFrame()

    for col_name in masked_df2:
        col1 = masked_df1[col_name]
        col2 = masked_df2[col_name]

        is_numerical_col = all(np.issubdtype(col.dtype, np.number) for col in [col1, col2])

        dfs = [diff_df, col1]

        if is_numerical_col:
            if atol and (abs(col1 - col2) < atol).all():
                #skip too small diff
                pass
            else:
                diff = col2 - col1
                diff_col = f'_diff_{len(diff_cols)}'
                diff = diff.rename(diff_col)
                dfs.append(col2)
                dfs.append(diff)
                diff_cols.append(diff_col)

        elif (col1 != col2).any():
            dfs.append(col2)

        diff_df = pd.concat(dfs, axis=1)

    with pd.option_context("display.max_rows", num_diffs_to_display):
        print_df = diff_df.copy()
        for col_name in diff_cols:
            print_df[col_name].replace(0, '', inplace=True)
        print_df.rename(columns=dict.fromkeys(diff_cols, 'diff'), inplace=True)
        if not print_df.empty:
            print()
            print(f'{msg} out of {num_diffs} {message} diffs:')

            print(getattr(print_df.reset_index(drop=False), func_name)(num_diffs_to_display))

    if diff_cols and print_largest:
        sorted_df = diff_df.fillna(0).reset_index(drop=False)
        if not sorted_df.empty:
            print()
            print("largest absolute diffs:")
            sorted_df = sorted_df.sort_values(by=diff_cols)
            for col_name in diff_cols:
                sorted_df[col_name].replace(0, '', inplace=True)

            sorted_df.rename(columns=dict.fromkeys(diff_cols, 'diff'), inplace=True)

            print(sorted_df.tail(num_diffs_to_display).reset_index(drop=True))
