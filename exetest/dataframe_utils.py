import numpy as np
import pandas as pd


def load_df(file_path, ignore_cols=None):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_feather(file_path)

    if ignore_cols:
        return df.loc[:, ~df.columns.isin(ignore_cols)]
    else:
        return df


class DFComparator:

    def __init__(self, ignore_cols=None, verbose: bool = True):
        self.ignore_cols = ignore_cols or []
        self.verbose = verbose

    def description(self) -> str:
        if self.ignore_cols:
            return f"ignoring columns: {self.ignore_cols}"
        else:
            return ''

    def __call__(self, df_path1, df_path2) -> bool:
        df1 = load_df(file_path=df_path1, ignore_cols=self.ignore_cols)
        df2 = load_df(file_path=df_path2, ignore_cols=self.ignore_cols)

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
                    if not np.isclose(df1[col].values, df2[col].values).all():
                        cols_with_diffs.append(col)
                else:
                    if not np.equal(df1[col].values, df2[col].values).all():
                        cols_with_diffs.append(col)

            if cols_with_diffs:
                if self.verbose:
                    print('====================================')
                    print(f'merge on cols with diff {cols_with_diffs}:')
                    df1_with_diff = df1[cols_with_diffs]
                    df2_with_diff = df2[cols_with_diffs]
                    merged_df = df1_with_diff.merge(df2_with_diff, indicator=True, how='outer')
                    print(merged_df[merged_df['_merge'] != 'both'])

                return False

        return True
