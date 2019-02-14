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


def default_file_diff(file_path1, file_path2, print_diff=True, ignore_lines=tuple(), max_diff_in_log=25):
    """ compare two files line by line:
        @:return: False if a difference is found
    """

    file1 = open(file_path1, 'r')
    file2 = open(file_path2, 'r')

    try:
        diff = difflib.unified_diff(file1.readlines(), file2.readlines(),
                                    fromfile=file_path1, tofile=file_path2,
                                    n=0  # no context lines
                                    )

        num_diffs = 0
        ignored_diffs = 0
        if diff:
            print()

        for diff_line in diff:
            if print_diff:
                print(diff_line, end='', file=sys.stderr)

            if any(diff_line.startswith(tag) for tag in ['---', '+++', '@@ ']):
                # skip diff output meta data
                continue
            else:

                ignore = False
                for file_name_ending in ignore_lines:
                    if file_path1.endswith(file_name_ending):
                        item = ignore_lines[file_name_ending]
                        # assume it's a list of patterns
                        ignore = item(diff_line) if callable(item) else any(line in diff_line for line in item)

                        if ignore:
                            ignored_diffs += 1
                            if diff_line.startswith('+'):
                                print('      Above diff is ignored', file=sys.stderr)
                                break

                if ignore:
                    continue

            num_diffs += 1
            if num_diffs >= max_diff_in_log:
                print(f'more than {num_diffs} diffs found - diff file manually for more details', file=sys.stderr)
                break

        if num_diffs == 0:
            if ignored_diffs == 0 and platform.system() != 'Windows':
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
