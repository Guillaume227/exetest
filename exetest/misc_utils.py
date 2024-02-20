import os
import sys
import shutil
import time
from contextlib import contextmanager
import subprocess
import typing
import pathlib


def print_test_meta(*message):
    print(*message, file=sys.stderr)


def rmdir(dir_path):
    if os.path.exists(os.path.join(os.getcwd(), dir_path)):
        shutil.rmtree(dir_path)


@contextmanager
def working_dir(path):
    """temporary switch of working directory"""

    current_dir = os.getcwd()
    if path is not None:
        os.chdir(path)

    try:
        yield
    finally:
        if path is not None:
            os.chdir(current_dir)


def prompt_user(message_prompt: str,
                yes_entries=('y',),
                no_entries=('n',),
                default: bool = None) -> bool:
    if default is not None:
        if default:
            reply_usage = f'[{yes_entries[0]}]/{no_entries[0]}'
        else:
            reply_usage = f'{yes_entries[0]}/[{no_entries[0]}]'
    else:
        reply_usage = f'{yes_entries[0]}/{no_entries[0]}'

    message_prompt += ' ' + reply_usage + '\n'

    while True:
        user_input = input(message_prompt)
        if user_input == '' and default is not None:
            return default
        if user_input in yes_entries:
            return True
        if user_input in no_entries:
            return False

        print(f'Invalid choice: {user_input}')


@contextmanager
def revert_file_modif(file_path):
    """ Saves file content and  write original content back on exit.
        Creates file if it does not exist and deletes it on exist.
    """

    file_exists = os.path.exists(file_path)
    with open(file_path) as f:
        original_content = f.read()

    try:
        yield original_content
    finally:
        if not file_exists:
            os.remove(file_path)
        else:
            with open(file_path, 'w') as f:
                f.write(original_content)


@contextmanager
def perf_timer(message='Time elapsed:', verbose=True):
    """timer context (includes time spent during process sleep)"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        if verbose:
            time_elapsed = time.perf_counter() - start_time
            print(message, '%.3fsec' % time_elapsed)


def diff_directory(path1, path2):
    """
    Iterate on dir tree below path2 and report if  a path under path2 does not exist under path2
    If some file or dir should be ignored, add them to ignored_dir_or_file.
    :param path1:
    :param path2:
    :return:
    """
    ignored_dir_or_file = ['.svn', '.git', '.gitignore', 'README.md']

    for path in os.listdir(path2):
        if path not in ignored_dir_or_file:
            abs1 = os.path.join(path1, path)
            abs2 = os.path.join(path2, path)

            if not os.path.exists(abs1):
                yield abs2

            if os.path.isdir(abs2):
                for diff_path in diff_directory(abs1, abs2):
                    yield diff_path


class ExeFailedException(Exception):
    pass


def format_log_for_exception(exc, max_num_lines=20):
    """
    extract information from streams captured by a subprocess error
    :param exc: an instance of subprocess.CalledProcessError
    :param max_num_lines: max number of log lines to append to reraised exception
    :return: formatted exception message with extra info from the captured executable log
    """

    exec_log = exc.stderr
    if not exec_log:
        exec_log = exc.stdout

    if exec_log:
        message = exec_log.decode('latin-1', errors='ignore')
        last_few_lines = os.linesep.join(message.rsplit(os.linesep, max_num_lines)[-(max_num_lines-1):])
        msg_len = len(message)

        if msg_len > len(last_few_lines):
            return last_few_lines + f'\n[truncated to last {max_num_lines} lines\n'
        else:
            return last_few_lines


def handle_subprocess_error(exc):
    message = format_log_for_exception(exc)
    if message:
        raise ExeFailedException(message) from exc


def exec_cmdline(command, args_list, check_ret_code=True,
                 log_save_path=None, env_vars=None,
                 pre_cmd=None, post_cmd=None, verbose=True, norun=False):
    """
    Runs command and returns the captured stdout log
    :param command:
    :param args_list:
    :param check_ret_code:
    :param log_save_path:
    :param env_vars:
    :param pre_cmd:
    :param post_cmd:
    :param verbose:
    :param norun:
    :return:
    """

    env_vars_line = ''
    if env_vars:
        env_vars_line = ' '.join([k + '=' + str(v) for k, v in env_vars.items()])
        env_vars_line += ' '

    if isinstance(args_list, str):
        args_list = [args_list]

    # get rid of new lines in command line args
    args_list = [' '.join(line.strip() for line in args.splitlines()) for args in args_list]

    cmd_line_tasks = [command + ' ' + args for args in args_list]

    # chain commands
    cmd_line = ''
    for task in cmd_line_tasks[:-1]:
        cmd_line += '{ ' + env_vars_line + task + ' & } ; '

    cmd_line += env_vars_line + cmd_line_tasks[-1]

    # add pre/post commands
    if pre_cmd:
        cmd_line = ' && '.join(pre_cmd) + ' && ' + cmd_line

    if post_cmd:
        cmd_line += ' && ' + ' && '.join(post_cmd)

    if verbose:
        print()
        print('working dir/command line:')
        print(os.getcwd())
        print(cmd_line)
        print()

    if not norun:
        if not os.path.exists(command):
            raise Exception(f'{command} '
                            f'not found from {os.getcwd()}. '
                            f'Is it in your path? Did you compile?')

        exe_name = os.path.basename(command.split()[0])
        try:
            with perf_timer(exe_name + ' execution took '):
                result_info = subprocess.run(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

            log_out = result_info.stdout
            if log_save_path is not None:
                # log_out = log_out.decode('latin-1')
                with open(log_save_path, 'wb') as f:
                    f.write(log_out)

            if check_ret_code:
                result_info.check_returncode()

            return log_out

        except subprocess.CalledProcessError as exc:
            handle_subprocess_error(exc)
            raise

    return cmd_line


def pattern_matches(pattern: typing.Union[str, typing.List[str]],
                    path_string: str) -> bool:

    if isinstance(pattern, str):
        patterns = [pattern]
    else:
        patterns = pattern

    has_suppress = False
    has_keep = False
    keep = None

    def match(pattern, string):
        if '*' in pattern:
            return pathlib.PurePath(string).match(pattern)
        else:
            return pattern in string

    for pattern in patterns:
        if pattern.startswith('~'):
            has_suppress = True
            if match(pattern=pattern[1:], string=path_string):
                keep = False

        else:
            has_keep = True
            if match(pattern=pattern, string=path_string):
                keep = True

    if keep is None:
        # default
        keep = has_suppress and not has_keep

    return keep
