#!/usr/bin/python3

import subprocess
import argparse
import os
import sys
from exetest import ExeTestEnvVars


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

    parser.add_argument("-j", "--num-cores",
                        help="number of CPUs used to run tests in parallel",
                        nargs='?',
                        default=-1,
                        type=int)

    parser.add_argument("--no-skip", "--ns",
                        help="force-run tests marked as skipped",
                        action="store_true")

    args, other_pytest_args = parser.parse_known_args()

    env_vars = {}
    command = ['pytest']
    verbose = False
    two_step_rebase = False
    rebase_arg = None

    if args.no_skip:
        env_vars[ExeTestEnvVars.DISABLE_SKIP] = ''

    if args.collect_only:
        command += ['--collect-only', '-q']

    else:
        env_vars[ExeTestEnvVars.NUM_DIFFS] = args.num_diffs

        if args.rebase is not None:
            items = args.rebase.split(":")
            two_step_rebase = len(items) > 1
            rebase_arg = f'{ExeTestEnvVars.REBASE}={items[0]} '

        if args.compare_only:
            env_vars[ExeTestEnvVars.COMPARE_ONLY] = ''
        if args.keep_output:
            env_vars[ExeTestEnvVars.KEEP_OUTPUT_ON_SUCCESS] = ''

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
        if num_cores < 0 and args.norun is None:
            if args.rebase is None:
                if len(args.test_cases) == 0:
                    # default number of cores for running all tests
                    num_cores = 32
                elif len(args.test_cases) > 1:
                    # sensible default: set the number of workers to be the number of arguments
                    # (although due to regex/filtering feature it might not match the actual number of tests to run)
                    num_cores = len(args.test_cases)

        if num_cores > 0:
            assert args.rebase is None or two_step_rebase, "rebase operation cannot be parallelized"
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
                    if 'or' not in args.test_cases:
                        join_string = ' or '
                    else:
                        join_string = ' '
                    command += ['-k', join_string.join(args.test_cases)]
                    break

            test_cases.append(test_case)
        else:
            command += [' '.join(test_cases)]

    if other_pytest_args:
        command += other_pytest_args

    if env_vars:
        env_var_str = [f'{k}={v}' for k, v in env_vars.items()]
        command = env_var_str + command

    if verbose:
        command_tokens = []
        for token in command:
            if ' ' in token:
                token = f'"{token}"'
            command_tokens.append(token)

    command_str = ' '.join(command)

    if rebase_arg is not None:
        if two_step_rebase:
            command_str_1 = f'{ExeTestEnvVars.KEEP_OUTPUT_ON_SUCCESS}=1 ' + command_str
        else:
            command_str_1 = rebase_arg + command_str
    else:
        command_str_1 = command_str

    if verbose:
        print('running:\n', command_str_1)
    proc = subprocess.run(command_str_1, shell=True, text=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if two_step_rebase:
        command_str_2 = rebase_arg + f'{ExeTestEnvVars.COMPARE_ONLY}=1 ' + command_str
        if verbose:
            print('running:\n', command_str_2)
        proc = subprocess.run(command_str_2, shell=True, text=True)
        sys.exit(proc.returncode)
