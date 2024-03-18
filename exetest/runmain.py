#!/usr/bin/python3

import subprocess
import argparse
import os
from exetest import ExeTestEnvVars
import exetest


def main(prog, description=''):
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)

    parser.add_argument("test_cases", nargs='*',
                        help="test cases to run - leave empty for running all tests",
                        default="")

    parser.add_argument("-k",
                        help="pytest -k expression for selecting tests",
                        default="")

    parser.add_argument("-r", "--rebase",
                        help="whether to prompt for test rebase in case of difference to reference output",
                        nargs='?',
                        default=None,
                        const='',
                        type=str)

    parser.add_argument("-f", "--file-filter",
                        help="filter files to rebase / compare",
                        nargs="+")

    parser.add_argument("--verbose",
                        help="verbose flag",
                        action='store_true')

    parser.add_argument("-c", "--compare-only",
                        help="run the compare step only (do not run the tests, assumes test outputs are present)",
                        action='store_true')

    parser.add_argument("-l", "--collect-only",
                        help="list test cases without running them",
                        action="store_true")

    parser.add_argument("--norun",
                        help="print test commands without running the tests",
                        nargs='?',
                        const='',
                        type=str)

    parser.add_argument("-d", "--num-diffs", "--diffs",
                        help="number of differing lines to display for a failed test",
                        nargs='?',
                        default=25,
                        type=int)

    parser.add_argument("--keep-output", "--ko",
                        help="Keep output files on success",
                        action="store_true")

    parser.add_argument("-t", "--show-timing",
                        help="show the N slowest tests",
                        nargs='?',
                        default=25,
                        type=int)

    parser.add_argument("-n", "--num-cores",
                        help="number of CPUs used to run tests in parallel",
                        nargs='?',
                        default=None,
                        type=int)

    parser.add_argument("--no-skip", "--ns",
                        help="force-run tests marked as skipped",
                        action="store_true")

    args, other_pytest_args = parser.parse_known_args()

    if args.rebase is not None and args.rebase != exetest.FORCE_REBASE and \
        (len(args.test_cases) != 1 or\
         (args.num_cores is not None and args.num_cores > 1)):
        # run two-step rebase:
        # 1. distribute the tests
        # 2. compare and prompt for rebase.

        rebase_arg = args.rebase

        if not args.compare_only:
            args.rebase = None
            ret_code = process_args(args, other_pytest_args)
            has_diff = ret_code != 0
            if not has_diff:
                return ret_code

        args.compare_only = True
        args.rebase = rebase_arg
        args.num_cores = 1
        other_pytest_args.append('--last-failed')
        return process_args(args, other_pytest_args)

    else:
        return process_args(args, other_pytest_args)


def process_args(args, other_pytest_args):
    env_vars = {}
    command = ['pytest']
    verbose = False

    if args.no_skip:
        env_vars[ExeTestEnvVars.DISABLE_SKIP] = ''

    if args.collect_only:
        command += ['--collect-only', '-q']

    else:
        env_vars[ExeTestEnvVars.NUM_DIFFS] = args.num_diffs

        if args.rebase is not None:
            env_vars[ExeTestEnvVars.REBASE] = args.rebase
        if args.compare_only:
            env_vars[ExeTestEnvVars.COMPARE_ONLY] = ''
        if args.file_filter:
            env_vars[ExeTestEnvVars.FILE_FILTER] = '+'.join(args.file_filter)

        command += ['-v']
        verbose = len(args.test_cases) > 0 or args.verbose

        if verbose:
            env_vars[ExeTestEnvVars.VERBOSE] = ''
            command += ['--tb=short']
        else:
            command += ['--tb=line']
            # command += ['--tb=no']

        if args.norun:
            env_vars[ExeTestEnvVars.NO_RUN] = args.norun

        num_cores = args.num_cores
        if num_cores is None and args.norun is None:
            if len(args.test_cases) == 0:
                # default number of cores for running all tests
                num_cores = 32
            elif len(args.test_cases) > 1:
                # sensible default: set the number of workers to be the number of arguments
                # (although due to regex/filtering feature it might not match the actual number of tests to run)
                num_cores = len(args.test_cases)

        if num_cores is not None and num_cores > 1:
            assert args.rebase is None or args.rebase == exetest.FORCE_REBASE, "rebase operation cannot be parallelized"
            command += ['-n', str(num_cores)]
        else:
            if verbose or args.rebase is not None:
                command += ['--capture=no']  # capture=no is incompatible with distributing tests

        if args.show_timing >= 0 and args.norun is None and not args.collect_only:
            command += [f'--durations={args.show_timing}']

    if args.test_cases:
        test_cases = []
        for test_case in args.test_cases:
            if not os.path.exists(test_case):
                if '::' not in test_case:
                    if 'or' not in args.test_cases and 'not' not in args.test_cases:
                        join_string = ' or '
                    else:
                        join_string = ' '
                    command += ['-k', '"' + join_string.join(args.test_cases) + '"']
                    break

            test_cases.append(test_case)
        else:
            command += [' '.join(test_cases)]

    if other_pytest_args:
        command += other_pytest_args

    return run_command(command=command, env_vars=env_vars, verbose=verbose)


def run_command(command, env_vars, verbose=True):

    if env_vars:
        env_var_str = [f'{k}={v}' for k, v in env_vars.items()]
        command = env_var_str + command

    if verbose:
        command_tokens = []
        for token in command:
            if ' ' in token:
                token = f'"{token}"'
            command_tokens.append(token)
        print('running:\n', ' '.join(command_tokens))

    command_str = ' '.join(command)
    proc = subprocess.run(command_str, shell=True, text=True)
    return proc.returncode
