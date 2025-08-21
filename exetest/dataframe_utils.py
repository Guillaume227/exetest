import numpy as np
import polars as pl
from typing import List, Optional, Union


def load_df(file_path, ignore_cols=None, filter_cols=None):
    if file_path.endswith('.csv'):
        df = pl.read_csv(file_path)
    else:
        df = pl.read_ipc(file_path)  # Feather format in polars

    if filter_cols:
        df = df.select(filter_cols)

    if ignore_cols:
        return df.select([col for col in df.columns if col not in ignore_cols])
    else:
        return df


def is_close(a, b, **kwargs):
    return np.isclose(a=b, b=a, **kwargs)


def uncentered_correlation(df1: pl.DataFrame, df2: pl.DataFrame):
    """
    Compute the uncentered correlation (cosine similarity) between corresponding columns of two DataFrames.
    """
    assert df1.shape == df2.shape, "DataFrames must have the same shape"

    # Convert to numpy for numerical operations
    df1_np = df1.select(df1.columns).to_numpy()
    df2_np = df2.select(df2.columns).to_numpy()

    # Normalize the dataframes
    df1_norm = np.linalg.norm(df1_np, axis=0)
    df2_norm = np.linalg.norm(df2_np, axis=0)

    # Compute the dot product
    dot_product = np.sum(df1_np * df2_np, axis=0)

    # Compute the uncentered correlation
    uncentered_corr = dot_product / (df1_norm * df2_norm)

    return pl.Series('corr', uncentered_corr)


def pearson_correlation(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    assert df1.shape == df2.shape, "Dataframes must have the same shape"

    numerical_cols = [
        col for col in df1.columns
        if col in df2.columns and is_numeric_type(df1.schema[col]) and is_numeric_type(df2.schema[col])
    ]

    if not numerical_cols:
        raise ValueError("No matching numeric columns to compute correlations")

    correlations = []
    for col in numerical_cols:
        temp_df = pl.DataFrame({f"{col}_df1": df1[col], f"{col}_df2": df2[col]})
        correlation_value = temp_df.select(
            pl.corr(f"{col}_df1", f"{col}_df2")
        )[0, 0]
        correlations.append(dict(Column=col, Correlation=correlation_value))

    return pl.DataFrame(correlations)


def is_float_type(pl_dtype) -> bool:
    return pl_dtype in [pl.Float64, pl.Float32]


def is_integer_type(pl_dtype) -> bool:
    return pl_dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]


def is_numeric_type(pl_dtype) -> bool:
    return is_float_type(pl_dtype) or is_integer_type(pl_dtype)


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
        self.np_close_kwargs.setdefault('equal_nan', True)
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

    def compare_dataframes(
            self,
            df1: Union[pl.DataFrame, pl.LazyFrame],
            df2: Union[pl.DataFrame, pl.LazyFrame],
            match_columns: bool = True,
            include_cols: Optional[List[str]] = None,
            exclude_cols: Optional[List[str]] = None,
            show_context_cols: bool = True) -> np.array:
        """
        :return: mask of rows with differences (True if row is the same)
        """

        df1 = df1 if isinstance(df1, pl.LazyFrame) else df1.lazy()
        df2 = df2 if isinstance(df2, pl.LazyFrame) else df2.lazy()

        exclude_cols = exclude_cols or []

        if self.filter_cols:
            df1 = df1.select(self.filter_cols)
            df2 = df2.select(self.filter_cols)

        df1_schema = df1.collect_schema()
        df2_schema = df2.collect_schema()
        df1_cols = set(df1_schema.names())
        df2_cols = set(df2_schema.names())

        df1_count = df1.select(pl.len()).collect().item()
        df2_count = df2.select(pl.len()).collect().item()

        if df1_cols == df2_cols and df1_count == df2_count:
            shape_differs = False
            try:
                hash1 = df1.select(pl.concat_str(pl.all()).hash()).sum().collect().item()
                hash2 = df2.select(pl.concat_str(pl.all()).hash()).sum().collect().item()

                if hash1 == hash2:
                    return np.array([True] * df1_count)

            except Exception:
                identical = df1.collect().equals(df2.collect())
                if identical:
                    return np.array([True] * df1_count)
        else:
            shape_differs = True

        if shape_differs and self.verbose:
            print(f'df1 shape: ({df1_count}, {len(df1_cols)})')
            print(f'df2 shape: ({df2_count}, {len(df2_cols)})')

        columns_differ = False
        df1_only_cols = df1_cols - df2_cols
        df2_only_cols = df2_cols - df1_cols

        if include_cols is not None:
            include_cols_set = set(include_cols)
            df1_only_cols = df1_only_cols.intersection(include_cols_set)
            df2_only_cols = df2_only_cols.intersection(include_cols_set)

        df1_only_cols = df1_only_cols - set(exclude_cols)
        df2_only_cols = df2_only_cols - set(exclude_cols)

        if df1_only_cols:
            columns_differ = True
            message = f'{len(df1_only_cols)} cols only in df1: {df1_only_cols}'
            if match_columns:
                raise ValueError(message)
            elif self.verbose:
                print(message)

        if df2_only_cols:
            columns_differ = True
            message = f'{len(df2_only_cols)} cols only in df2: {df2_only_cols}'
            if match_columns:
                raise ValueError(message)
            elif self.verbose:
                print(message)

        if include_cols is None:
            include_cols_set = df1_cols.union(df2_cols)
        else:
            include_cols_set = set(include_cols)

        num_row_difference = 0
        common_cols = list(df1_cols.intersection(df2_cols) - set(exclude_cols))

        if shape_differs or columns_differ:
            if self.verbose:
                if len(common_cols) != len(df1_cols) or len(common_cols) != len(df2_cols):
                    print(len(common_cols), 'common cols:', common_cols)
                    print()

            if not common_cols:
                # all rows differ, no common cols
                return np.array([False] * df1.height)

            df1 = df1.select(common_cols)
            df2 = df2.select(common_cols)

            if df1_count != df2_count:
                num_row_difference = abs(df1_count - df2_count)
                print(f"number of rows differ: {df1_count} vs {df2_count}")
                min_rows = min(df1_count, df2_count)
                df1 = df1.slice(0, min_rows)
                df2 = df2.slice(0, min_rows)

        # Replace infinities with nulls
        inf_expr = []
        for col in common_cols:
            col_dtype = df1.collect_schema()[col]
            if col_dtype in [pl.Float32, pl.Float64]:
                inf_expr.append(
                    pl.when(pl.col(col).is_infinite())
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        if inf_expr:
            df1 = df1.with_columns(inf_expr)
            df2 = df2.with_columns(inf_expr)

        df1 = df1.collect()
        df2 = df2.collect()

        rows_count = df1.height
        diff_mask = np.zeros(rows_count, dtype = bool)

        differing_nan_cols = []
        nan_row_mask = np.zeros(rows_count, dtype=bool)

        for col in common_cols:
            col1_nans = df1[col].is_null().to_numpy()
            col2_nans = df2[col].is_null().to_numpy()

            col_diff = np.logical_xor(col1_nans, col2_nans)

            if col_diff.any():
                differing_nan_cols.append(col)
                nan_row_mask = np.logical_or(nan_row_mask, col_diff)

        diff_mask = nan_row_mask.copy()

        if differing_nan_cols:
            print(len(differing_nan_cols), 'cols with nan differences:', ' '.join(differing_nan_cols))
            if self.verbose and self.num_diffs_to_display:
                print_df_diff(
                    df1 if show_context_cols else df1.select(differing_nan_cols),
                    df2 if show_context_cols else df2.select(differing_nan_cols),
                    diff_mask = nan_row_mask,
                    num_diffs_to_display=self.num_diffs_to_display,
                    message = 'nans',
                    print_largest=False
                )
        else:
            print('no cols with Nans difference')

        matching_nan_mask = np.zeros(rows_count, dtype=bool)
        for col in common_cols:
            col1_nans = df1[col].is_null().to_numpy()
            col2_nans = df2[col].is_null().to_numpy()
            matching_nan_mask = np.logical_or(matching_nan_mask, np.logical_and(col1_nans, col2_nans))

        num_matching_nans = matching_nan_mask.sum()
        if self.verbose and num_matching_nans:
            print("there are", num_matching_nans, "matching nans")

        numerical_diff_cols = []
        non_numerical_diff_cols = []

        for col in common_cols:
            if col not in include_cols_set or col in exclude_cols:
                continue

            df1_col_type = df1_schema[col]
            df2_col_type = df2_schema[col]

            col1_nans = df1[col].is_null().to_numpy()
            col2_nans = df2[col].is_null().to_numpy()
            matching_nans = np.logical_and(col1_nans, col2_nans)

            col1_vals = df1[col].to_numpy()
            col2_vals = df2[col].to_numpy()

            valid_positions = ~matching_nans

            is_float_col = is_float_type(df2_col_type) and is_float_type(df2_col_type)
            is_integer_col = is_integer_type(df2_col_type) and is_integer_type(df2_col_type)

            is_numerical_col = is_numeric_type(df1_col_type) and is_numeric_type(df2_col_type)

            valid_mask = ~(col1_nans | col2_nans)
            if valid_mask.any():

                if is_float_col and not is_integer_col:
                    comparison = np.zeros_like(valid_mask, dtype=bool)
                    comparison[valid_mask] = np.isclose(
                        col1_vals[valid_mask],
                        col2_vals[valid_mask],
                        **self.np_close_kwargs
                    )
                    if not comparison[valid_mask].all():
                        numerical_diff_cols.append(col)

                elif is_numerical_col:
                    if not np.array_equal(col2_vals[valid_mask], col2_vals[valid_mask]):
                        numerical_diff_cols.append(col)

                else:
                    if not np.array_equal(col1_vals[valid_mask], col2_vals[valid_mask]):
                        non_numerical_diff_cols.append(col)

        if numerical_diff_cols or non_numerical_diff_cols:
            if numerical_diff_cols and self.verbose:
                single_value_cols = []
                correlation_cols = []

                for col in numerical_diff_cols:
                    is_unique_1 = df1.select(pl.col(col)).n_unique() <= 1
                    is_unique_2 = df2.select(pl.col(col)).n_unique() <= 1
                    if is_unique_1 or is_unique_2:
                        single_value_cols.append(col)
                    else:
                        correlation_cols.append(col)

                with pl.Config(float_precision=None):
                    print()
                    print(f'correlation of numerical cols ({rows_count} rows):')

                    if single_value_cols:
                        print("Cannot compute corr for cols with contstant values:", " ".join(single_value_cols))
                    if correlation_cols:
                        num_diff_df1 = df1.select(correlation_cols)
                        num_diff_df2 = df2.select(correlation_cols)

                        uncentered_corrs = uncentered_correlation(num_diff_df1, num_diff_df2)
                        centered_corrs = pearson_correlation(num_diff_df1, num_diff_df2)

                        corrs_df = centered_corrs.join(uncentered_corrs, how='full', on='Column', coalesce=True)

                        sorted_corrs = (corrs_df
                                        .filter(~pl.all_horizontal(pl.col("*").is_null()))
                                        .sort("Correlation", descending=True))

                        pandas_print(sorted_corrs)

                        if len(correlation_cols) > 16:
                            print(f'(on {rows_count} rows)')
                        print()

            if numerical_diff_cols:
                diff_mask_numerical = np.zeros(rows_count, dtype=bool)

                for col in numerical_diff_cols:
                    col1_vals = df1[col].to_numpy()
                    col2_vals = df2[col].to_numpy()
                    col1_nans = df1[col].is_null().to_numpy()
                    col2_nans = df2[col].is_null().to_numpy()

                    valid_mask = ~(col1_nans | col2_nans)

                    if valid_mask.any():
                        comparison = np.zeros_like(valid_mask, dtype=bool)
                        comparison[valid_mask] = ~np.isclose(
                            col1_vals[valid_mask],
                            col2_vals[valid_mask],
                            **self.np_close_kwargs
                        )
                        diff_mask_numerical = np.logical_or(diff_mask_numerical, comparison)

                diff_mask = np.logical_or(diff_mask, diff_mask_numerical)

                if self.verbose:
                    with pl.Config(float_precision=None):
                        df1_with_diff = df1.select(numerical_diff_cols if not show_context_cols else df1.columns)
                        df2_with_diff = df2.select(numerical_diff_cols if not show_context_cols else df2.columns)

                        print_df_diff(
                            df1_with_diff,
                            df2_with_diff,
                            diff_mask = diff_mask_numerical,
                            num_diffs_to_display=self.num_diffs_to_display,
                            message='numerical'
                        )
        if non_numerical_diff_cols:
            diff_mask_non_numerical = np.zeros(rows_count, dtype=bool)

            for col in non_numerical_diff_cols:
                col1_vals = df1[col].to_numpy()
                col2_vals = df2[col].to_numpy()
                col1_nans = df1[col].is_null().to_numpy()
                col2_nans = df2[col].is_null().to_numpy()

                valid_mask = ~(col1_nans | col2_nans)

                if valid_mask.any():
                    comparison = (col1_vals != col2_vals) & valid_mask
                    diff_mask_non_numerical = np.logical_or(diff_mask_non_numerical, comparison)

            diff_mask = np.logical_or(diff_mask, diff_mask_non_numerical)

            if self.verbose and self.num_diffs_to_display:
                df1_with_diff = df1.select(numerical_diff_cols if not show_context_cols else df1.columns)
                df2_with_diff = df2.select(numerical_diff_cols if not show_context_cols else df2.columns)

                print_df_diff(
                    df1_with_diff,
                    df2_with_diff,
                    diff_mask = diff_mask_non_numerical,
                    num_diffs_to_display=self.num_diffs_to_display,
                    message='non-numerical'
                )
        if num_row_difference:
            extra_false = np.full(num_row_difference, True, dtype=bool)
            diff_mask = np.concatenate((diff_mask, extra_false))

        return ~diff_mask


def to_df(ldf: Union[pl.LazyFrame, pl.DataFrame]) -> pl.DataFrame:
    return ldf.collect() if isinstance(ldf, pl.LazyFrame) else ldf


def to_pandas(df, fillna: bool = False):
    df = to_df(df)
    time_cols = [col for col, dtype in df.schema.items() if dtype == pl.Time]

    for time_col in time_cols:
        df = df.with_columns(
            pl.datetime(2000, 1, 1).dt.combine(pl.col(time_col)).cast(pl.Datetime("us")).dt.time().alias(time_col)
        )

    if fillna:
        for col, dtype in df.schema.items():
            if dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(col).fill_null("").alias(col)
                )
    pd_df = df.to_pandas()
    return pd_df


def pandas_print(df, index: bool = False, fillna: bool = False, col_rename=None):
    pd_df = to_pandas(df, fillna==fillna)
    if fillna:
        pd_df = pd_df.fillna('')

    for col, dtype in df.collect_schema().items():
        if dtype == pl.Time:
            pd_df[col] = pd_df[col].astype(str).str.replace(r"^00:00:0|^00:00:|^00:", '', regex=True)
        elif dtype == pl.Duration:
            pd_df[col] = pd_df[col].astype(str).str.replace("0 days ", '')

    if col_rename:
        pd_df = pd_df.rename(columns=col_rename)

    print(pd_df.to_string(index=index))


def print_df_diff(df1: Union[pl.DataFrame, pl.LazyFrame],
                  df2: Union[pl.DataFrame, pl.LazyFrame],
                  diff_mask: np.ndarray,
                  num_diffs_to_display: int,
                  message: str,
                  atol: float = None,
                  print_largest: bool = True):
    """
    Print differences between two polars DataFrames.
    """

    df1 = to_df(df1)
    df2 = to_df(df2)

    if num_diffs_to_display > 0:
        msg = 'first'
        slice_func = lambda df, n: df.slice(0, n)
    elif num_diffs_to_display < 0:
        msg = 'last'
        slice_func = lambda df, n: df.slice(-n, None)
    else:
        return

    num_diffs = int(diff_mask.sum())
    num_diffs_to_display = min(df1.height, abs(num_diffs_to_display), num_diffs)
    msg += f' {num_diffs_to_display} differing rows'

    # Filter rows with differences
    masked_df1 = df1.filter(pl.Series(diff_mask))
    masked_df2 = df2.filter(pl.Series(diff_mask))

    result_data = dict(index=pl.arange(0, masked_df1.height, eager=True))
    diff_cols = []

    for col_name in masked_df2.columns:
        if col_name not in masked_df1.columns:
            continue

        col1 = masked_df1[col_name]
        col2 = masked_df2[col_name]

        is_numerical_col = is_numeric_type(col1.dtype) and is_numeric_type(col2.dtype)

        if is_numerical_col:
            diff = col2 - col1

            if atol is not None and (diff.abs() < atol).all():
                continue

            result_data[f"{col_name}"] = col1
            result_data[f"{col_name}'"] = col2

            diff_col = f'_diff_{len(diff_cols)}'
            result_data[diff_col] = diff
            diff_cols.append(diff_col)
        elif (col1 != col2).any():
            result_data[f"{col_name}"] = col1
            result_data[f"{col_name}'"] = col2

    if len(result_data) <= 1:
        return

    diff_df = pl.DataFrame(result_data)

    display_exprs = []
    rename_map = {}

    for col in diff_df.columns:
        if col in diff_cols:
            expr = (pl.when*(pl.col(col)==0)
                    .then(pl.lit(""))
                    .otherwise(pl.col(col)))
            rename_map[col] = 'diff'
            display_exprs.append(expr.alias(col))
        else:
            display_exprs.append(pl.col(col))

    print_df = diff_df.select(display_exprs)

    if num_diffs > 0:
        print()
        print(f'{msg} out of {num_diffs} {message} diffs:')
        pandas_print(slice_func(print_df, num_diffs_to_display, col_rename=rename_map))

    if diff_cols and print_largest:
        print()
        print("largest absolute diffs:")

        lazy_sorting = diff_df.lazy().fill_null(0)

        abs_sort_keys = [pl.col(col).abs() for col in diff_cols]
        lazy_sorting = lazy_sorting.sort(by = abs_sort_keys, descending=True)

        sorted_df = lazy_sorting.collect()

        display_exprs = []
        for col in sorted_df.columns:
            if col in diff_cols:
                expr = (pl.when(pl.col(col) == 0)
                        .then(pl.lit(""))
                        .otherwise(pl.col(col)))
                display_exprs.append(expr.alias(col))
            else:
                display_exprs.append(pl.col(col))

        display_df = sorted_df.select(display_exprs)

        pandas_print(slice_func(display_df, num_diffs_to_display), col_rename=rename_map)