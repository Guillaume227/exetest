import numpy as np
import pandas as pd
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
        return self.compare_dataframes(df1, df2)

    def compare_dataframes(self, df1, df2):
        if not df1.equals(df2):

            shape_differs = df1.shape != df2.shape
            if shape_differs and self.verbose:
                print('df1 shape:', df1.shape)
                print('df2 shape:', df2.shape)

            columns_differ = False
            cols = df1.columns.difference(df2.columns).values
            if cols.any():
                columns_differ = True
                if self.verbose:
                    print(len(cols), 'cols only in df1:', cols)

            cols = df2.columns.difference(df1.columns).values
            if cols.any():
                columns_differ = True
                if self.verbose:
                    print(len(cols), 'cols only in df2:', cols)
                    print()

            if shape_differs or columns_differ:
                common_cols = df1.columns.intersection(df2.columns).values
                if self.verbose:
                    print(len(common_cols), 'common cols:', common_cols)
                    print()

                if common_cols.any():
                    df1 = df1.loc[:, common_cols]
                    df2 = df2.loc[:, common_cols]
                else:
                    return False

            for df in df1, df2:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)

            if self.verbose:
                df1_nans = df1.isna()
                df2_nans = df2.isna()
                differing_nan_mask = df1_nans ^ df2_nans

                nan_col_mask = differing_nan_mask.any(axis=0)

                if cols_with_nans := nan_col_mask[nan_col_mask].index.to_list():
                    nan_row_mask = differing_nan_mask.any(axis=1)
                    print(len(cols_with_nans), 'cols with nan differences:', cols_with_nans)

                    print_df_diff(df1[cols_with_nans],
                                  df2[cols_with_nans],
                                  diff_mask=nan_row_mask,
                                  num_diffs_to_display=self.num_diffs_to_display,
                                  message='nans')

            cols_with_diffs = []
            for col in df1.columns:
                if df1[col].dtype != 'category' and np.issubdtype(df1[col].dtype, np.number) \
                                                and np.issubdtype(df2[col].dtype, np.number):
                    # use numerical comparison
                    if not np.allclose(df1[col].values, df2[col].values, **self.np_close_kwargs):
                        cols_with_diffs.append(col)
                else:
                    if not np.equal(df1[col].values, df2[col].values).all():
                        cols_with_diffs.append(col)

            if cols_with_diffs:
                if self.verbose:
                    numerical_diff_cols = []
                    non_numerical_diff_cols = []
                    for col in cols_with_diffs:
                        if np.issubdtype(df1[col].dtype, np.number) and \
                           np.issubdtype(df2[col].dtype, np.number):
                            numerical_diff_cols.append(col)
                        else:
                            non_numerical_diff_cols.append(col)

                    if numerical_diff_cols:
                        with pd.option_context("display.float_format", "{:.2f}".format):
                            print(f'correlation of numerical cols ({df1.shape[0]} rows):')
                            corrs = df1[numerical_diff_cols].corrwith(df2[numerical_diff_cols])
                            print(corrs.sort_values(ascending=False).to_string())
                            if corrs.shape[0] > 16:
                                print(f'(on {df1.shape[0]} rows)')
                            print()

                    if self.num_diffs_to_display:
                        if numerical_diff_cols:

                            df1_with_diff = df1[numerical_diff_cols]
                            df2_with_diff = df2[numerical_diff_cols]
                            diff_mask = ~(df1_with_diff - df2_with_diff).apply(
                                functools.partial(is_close, b=0, **self.np_close_kwargs))
                            diff_mask = diff_mask.any(axis=1)

                            print_df_diff(df1_with_diff,
                                          df2_with_diff,
                                          diff_mask=diff_mask,
                                          num_diffs_to_display=self.num_diffs_to_display,
                                          message='numerical')

                        if non_numerical_diff_cols:
                            df1_with_diff = df1[non_numerical_diff_cols]
                            df2_with_diff = df2[non_numerical_diff_cols]
                            diff_mask = (df1_with_diff != df2_with_diff).any(axis=1)

                            print_df_diff(df1_with_diff,
                                          df2_with_diff,
                                          diff_mask=diff_mask,
                                          num_diffs_to_display=self.num_diffs_to_display,
                                          message='non-numerical')

                return False

        return True


def print_df_diff(df1, df2, diff_mask, num_diffs_to_display, message):
    if num_diffs_to_display > 0:
        msg = f'first {num_diffs_to_display} differing rows'
        func_name = 'head'
    else:
        msg = f'last {abs(num_diffs_to_display)} differing rows'
        func_name = 'tail'

    masked_df1 = getattr(df1.reset_index()[diff_mask], func_name)(abs(num_diffs_to_display))
    masked_df2 = getattr(df2[diff_mask], func_name)(abs(num_diffs_to_display))

    print(f'{msg} {diff_mask.sum()}/{diff_mask.shape[0]} {message} diffs:')

    diff_df = pd.DataFrame(masked_df1['index'])
    for col_name in masked_df2:
        diff_df = pd.concat([diff_df, masked_df1[col_name], masked_df2[col_name]], axis=1)

    with pd.option_context("display.max_rows", abs(num_diffs_to_display)):
        print(diff_df.reset_index(drop=True))
