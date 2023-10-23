import filecmp
import os
import os.path
import platform
import sys
import difflib

# import subprocess
# import numpy as np
# import pandas as pd


globalIgnoreLines = {}


class FilteredFileReader:
    """
    a file reader that optionally selects lines containing a substring
    """

    def __init__(self, file_path, filter_tag=None, skip_lines=None):
        """

        :param file_path:
        :param filter_tag: keep only lines containing that substring
        :param skip_lines: list of line indexes to keep
        """
        self._file = open(file_path, 'r')
        self._filter_tag = filter_tag
        self._skip_lines = skip_lines

    def readlines(self):

        if self._filter_tag is None:
            lines = self._file.readlines()
        else:
            lines = [line.split(self._filter_tag)[-1] for line in self._file.readlines()]

        if self._skip_lines is None:
            return lines
        else:
            return [line for i, line in enumerate(lines) if i not in self._skip_lines]


def default_file_diff(file_path1, file_path2,
                      print_diff=True,
                      ignore_patterns=tuple(),
                      filter_tag=None,
                      skip_lines=None,
                      max_diff_in_log=25):
    """ compare two files line by line:
        @:return: False if a difference is found
    """

    try:
        filter_tag1, filter_tag2 = filter_tag
    except:
        filter_tag1 = filter_tag2 = filter_tag

    file1 = FilteredFileReader(file_path1, filter_tag=filter_tag1, skip_lines=skip_lines)
    file2 = FilteredFileReader(file_path2, filter_tag=filter_tag2, skip_lines=skip_lines)

    try:
        diff = difflib.unified_diff(file1.readlines(), file2.readlines(),
                                    # fromfile=file_path1, tofile=file_path2,
                                    n=0  # no context lines
                                    )

        num_diffs = 0
        ignored_diffs = 0
        first_diff = True

        for diff_line in diff:

            if first_diff:
                first_diff = False
                print()

            if any(diff_line.startswith(tag) for tag in ['---', '+++', '@@ ']):
                # skip diff output meta data
                continue
            else:

                ignore = False
                for ignore_pattern in ignore_patterns:
                    # assume it's a list of patterns
                    if callable(ignore_pattern):
                        ignore = ignore_pattern(diff_line)
                    else:
                        ignore = ignore_pattern in diff_line

                    if ignore:
                        ignored_diffs += 1
                        break

                if ignore:
                    continue

            if print_diff:
                print(diff_line, end='', file=sys.stderr)

            num_diffs += 1

            if max_diff_in_log == 0:
                return False

            if num_diffs >= max_diff_in_log:
                print(f'Only showing {max_diff_in_log} differences out of {num_diffs} '
                      f'- diff file manually for more details', file=sys.stderr)
                break

        if num_diffs == 0:
            if ignored_diffs > 0 and platform.system() != 'Windows':
                print('        Ignoring Diff on {0} coming from line endings unix vs windows'.format(file_path1))

            return True

        return False

    except UnicodeDecodeError:
        print("     skipping text diff on non-text file:", file_path1)
        return False


def diff_dirs(ref_dir, test_dir, print_diff=True, ignore_lines=None, comparators=None, ignore_missing=False):
    """
    recursive directory compare:
    :param ref_dir:
    :param test_dir:
    :param print_diff:
    :param ignore_lines: {file_ext: [ string1, string2, ... ]}
        where the differing line will be ignored if it contains any of the strings
    :param comparators: {file_ext: comparator_func}
        where comparator_func: f(file1, file2) -> bool (True if no diff)
    :param ignore_missing:
    :return: differing files in @ref_dir vs @test_dir
    """

    dir_comp_res = filecmp.dircmp(ref_dir, test_dir)

    # print out what was found in each directory for information
    if print_diff:
        print()
        dir_comp_res.report()
        print()

    if not ignore_missing:
        # check number of files is the same
        assert len(dir_comp_res.right_only) == 0, f'Files only in {dir_comp_res.right}: {dir_comp_res.right_only}'
        assert len(dir_comp_res.left_only) == 0, f'Files only in {dir_comp_res.left}: {dir_comp_res.left_only}'

        # check names of files are the same
        assert set(dir_comp_res.right_list).issuperset(dir_comp_res.left_list), \
            f'Different file names found {set(dir_comp_res.right_list).difference(dir_comp_res.left_list)}'

    for diff_file in dir_comp_res.diff_files:

        file_name1 = os.path.join(ref_dir, diff_file)
        file_name2 = os.path.join(test_dir, diff_file)

        if ignore_missing:
            if not os.path.exists(file_name1):
                missing = file_name1
            elif not os.path.exists(file_name2):
                missing = file_name2
            else:
                missing = None

            if missing:
                print(f'missing directory', missing, file=sys.stderr)
                continue

        found_comparator = False
        # use a custom comparator fun if it exists for that file extension
        for comparator_dict in [comparators, globalIgnoreLines]:
            for file_ext in comparator_dict:
                if diff_file.endswith(file_ext):
                    found_comparator = True

                    if comparator_dict[file_ext](file_name1, file_name2):
                        yield diff_file

                    break

        if not found_comparator:
            if default_file_diff(file_name1, file_name2, print_diff, ignore_lines):
                yield diff_file

    for file_name in os.listdir(ref_dir):
        if os.path.isdir(os.path.join(ref_dir, file_name)):
            dir_path1 = os.path.join(ref_dir, file_name)
            dir_path2 = os.path.join(test_dir, file_name)

            if ignore_missing:
                if not os.path.exists(dir_path1):
                    missing = dir_path1
                elif not os.path.exists(dir_path2):
                    missing = dir_path2
                else:
                    missing = None

                if missing:
                    print('Missing directory', missing, file=sys.stderr)
                    continue

            for diff_file in diff_dirs(dir_path1, dir_path2, print_diff=print_diff,
                                       ignore_lines=ignore_lines,
                                       comparators=comparators,
                                       ignore_missing=ignore_missing):
                yield os.path.join(file_name, diff_file)


class FileComparator:

    def __init__(self, include_line_if_contains=None, skip_lines=None, **kwargs):

        self.include_line_if_contains = include_line_if_contains
        self.skip_lines = skip_lines
        self.kwargs = kwargs

    def description(self):
        if self.include_line_if_contains:
            if isinstance(self.include_line_if_contains, str):
                include_str = f'"{self.include_line_if_contains}"'
            else:
                include_str = ", ".join(f'"{line}"' for line in self.include_line_if_contains if line is not None)
            return f'filter lines on {include_str}'
        return ''

    def __call__(self, *args, **kwargs):
        return default_file_diff(*args, **kwargs,
                                 filter_tag=self.include_line_if_contains,
                                 skip_lines=self.skip_lines,
                                 **self.kwargs)
