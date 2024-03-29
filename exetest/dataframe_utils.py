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
                 num_diffs: int=10,
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
        self.num_diffs = num_diffs

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
                    print('cols only in df1:', cols)

            cols = df2.columns.difference(df1.columns).values
            if cols.any():
                columns_differ = True
                if self.verbose:
                    print('cols only in df2:', cols)

            if shape_differs or columns_differ:
                return False

            cols_with_diffs = []
            for col in df1.columns:
                if df1[col].dtype != 'category' and np.issubdtype(df1[col].dtype, np.number):
                    # use numerical comparison
                    if not np.allclose(df1[col].values, df2[col].values, **self.np_close_kwargs):
                        cols_with_diffs.append(col)
                else:
                    if not np.equal(df1[col].values, df2[col].values).all():
                        cols_with_diffs.append(col)

            if cols_with_diffs:
                if self.verbose:
                    print('====================================')
                    print(f'Showing first {self.num_diffs} in cols with diff {cols_with_diffs}:')
                    numerical_diff_cols =[]
                    non_numerical_diff_cols = []
                    for col in cols_with_diffs:
                        if np.issubdtype(df1[col].dtype, np.number):
                            numerical_diff_cols.append(col)
                        else:
                            non_numerical_diff_cols.append(col)

                    if numerical_diff_cols:
                        print('numerical diffs:')
                        df1_with_diff = df1[numerical_diff_cols]
                        df2_with_diff = df2[numerical_diff_cols]
                        diff_mask = ~(df1_with_diff - df2_with_diff).apply(
                            functools.partial(is_close, b=0, **self.np_close_kwargs))
                        print(pd.concat([df1_with_diff[diff_mask],
                                         df2_with_diff[diff_mask]
                                         ], axis=1).head(self.num_diffs))

                    if non_numerical_diff_cols:
                        print('non numerical diffs:')
                        df1_with_diff = df1[non_numerical_diff_cols]
                        df2_with_diff = df2[non_numerical_diff_cols]
                        diff_mask = df1_with_diff != df2_with_diff
                        print(pd.concat([df1_with_diff[diff_mask],
                                         df2_with_diff[diff_mask]
                                         ], axis=1).head(self.num_diffs))
                return False

        return True
