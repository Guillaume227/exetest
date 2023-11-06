import os
import os.path
from . import misc_utils
from .misc_utils import working_dir, rmdir
from .diff_utils import FileComparator
from functools import wraps
import shutil
import sys
import unittest
from .env_vars import ExeTestEnvVars
import pytest


force_rebase = 'FORCE'


def skip_test(reason=''):
    """
    A decorator for marking tests as skipped and still allowing to force-run them.
    Bypassing behavior is triggered by the presence of an environment variable.
    :param reason: why the test is being skipped
    """
    if ExeTestEnvVars.DISABLE_SKIP in os.environ:
        return lambda x: x
    else:
        return pytest.mark.skip(reason=reason)


def skip_module(reason=''):
    if ExeTestEnvVars.DISABLE_SKIP not in os.environ:
        pytest.skip(reason, allow_module_level=True)


class ExeTestCaseDecorator:
    """
    A test case decorator for testing an executable outputs
    by comparing new output to reference output
    """

    def __init__(self,
                 exe,
                 test_root='.',  # =os.path.dirname(__file__),
                 ref_dir='ref_dir',
                 out_dir='out_dir',
                 exe_args=None,
                 compare_spec=None,
                 run_from_out_dir=True,
                 test_name_as_dir=True,
                 nested_ref_dir=True,
                 nested_out_dir=False,
                 comparators=None,
                 exception_handler=None,
                 keep_output_on_success=False,
                 env_vars=None,
                 pre_cmd=None,
                 exe_path_ref=None,
                 log_output_path=None):
        """

        :param exe: path to the executable to test
        :param test_root: working directory for test execution from which ref_dir/out_dir paths are defined if relative.
        :param ref_dir: absolute path or relative to test_root directory where test case is defined
        :param out_dir: directory to write test output to
        :param exe_args: test executable arguments in string format - as would be passed to command line
        :param run_from_out_dir: whether the executable working directory should be the output directory
        :param test_name_as_dir: whether the test name should be used to infer the ref and output directories
        :param nested_ref_dir: whether ref_dir should be nested in the test_name directory (otherwise, test_name is nested under ref_dir)
        :param nested_out_dir: whether out_dir should be nested in the test_name directory (otherwise, test_name is nested under out_dir)
        :param comparators:
        :param exception_handler: what to do in case of exception
        :param env_vars: environment variables to populate for the test run
        :param pre_cmd: a command to run before the executable is run
        :param exe_path_ref:
        :param log_output_path:
        """

        self.exe_path_ref = exe_path_ref if exe_path_ref else {}

        def get_test_exe(val):
            if val in self.exe_path_ref:
                return self.exe_path_ref[val]
            elif val:
                if os.path.isdir(val):
                    return os.path.join(val, os.path.basename(exe))
                else:
                    return val
            else:
                return exe

        use_test_exe = os.environ.get(ExeTestEnvVars.USE_EXE)

        # if USE_TEST_EXE has special target value, run against reference executable
        if use_test_exe is not None:
            self.exe_path = get_test_exe(use_test_exe)
        else:
            self.exe_path = exe

        self.test_root = test_root
        self.REF_OUTPUT_DIR = ref_dir
        self.TMP_OUTPUT_DIR = out_dir
        self.test_name_as_dir = test_name_as_dir
        self.run_from_out_dir = run_from_out_dir

        self.test_name = ''
        self.comparators = comparators if comparators is not None else {}
        self.exception_handler = exception_handler
        self.common_env_vars = env_vars or dict()
        self.pre_cmd = pre_cmd if pre_cmd else []
        self.exe_args = exe_args
        self.log_output_path = log_output_path

        self.verbose = ExeTestEnvVars.VERBOSE in os.environ
        if ExeTestEnvVars.NO_RUN in os.environ:
            self.norun = os.environ[ExeTestEnvVars.NO_RUN]
            if not self.norun:
                self.norun = 'default'
        else:
            self.norun = None

        self._compare_spec = compare_spec
        self._nested_ref_dir = nested_ref_dir
        self._nested_out_dir = nested_out_dir
        self._keep_output_on_success = True if ExeTestEnvVars.KEEP_OUTPUT_ON_SUCCESS in os.environ else keep_output_on_success
        self._num_lines_diff = int(os.environ.get(ExeTestEnvVars.NUM_DIFFS, 20))

    @staticmethod
    def get_test_subdir(test_name):
        return test_name

    def get_output_dir(self, test_name):
        return self.make_dir_path(self.TMP_OUTPUT_DIR, test_name, self._nested_out_dir)

    def get_ref_dir(self, test_name):
        return self.make_dir_path(self.REF_OUTPUT_DIR, test_name, self._nested_ref_dir)

    def __call__(self,
                 exe_args=None,
                 extra_args=None,
                 compare_spec=None,
                 pre_cmd=None,
                 env_vars=None,
                 post_cmd=None,
                 owners=None):
        """

        :param exe_args: overrides arguments specified in the constructor
        :param extra_args: appends arguments to the ones specified in the constructor
        :param compare_spec:
        :param pre_cmd:
        :param env_vars:
        :param post_cmd:
        :param owners:
        :return:
        """
        all_env_vars = dict(self.common_env_vars)
        if env_vars:
            all_env_vars.update(env_vars)

        pre_cmds = self.pre_cmd
        if pre_cmd:
            if isinstance(pre_cmd, str):
                pre_cmds = pre_cmds + [pre_cmd]
            else:
                pre_cmds = pre_cmds + pre_cmd

        if exe_args is None:
            if self.exe_args is not None:
                exe_args = self.exe_args
            else:
                exe_args = ''

        if extra_args is not None:
            exe_args += ' ' + extra_args

        if compare_spec is None:
            compare_spec = self._compare_spec

        self._compare_only = ExeTestEnvVars.COMPARE_ONLY in os.environ

        def func_wrapper(test_func):
            """
            :param test_func:
            :return: decorated function
            """
            test_name = test_func.__name__.split("_", 1)[-1]
            import inspect
            parent_module_name = inspect.getmodule(test_func).__name__

            resolved_env_vars = {}
            for name, value in all_env_vars.items():
                resolved_env_vars[name] = value.format(test_name=test_name)

            @wraps(test_func)
            def f(*args, **kwargs):
                ret = test_func(*args, **kwargs)
                try:
                    self.run_test(exe_args,
                                  compare_spec,
                                  pre_cmd=pre_cmds,
                                  env_vars=resolved_env_vars,
                                  post_cmd=post_cmd,
                                  test_name=test_name,
                                  parent_module_name=parent_module_name,
                                  verbose=self.verbose)
                except unittest.SkipTest:
                    raise

                return ret

            def doc_gist(func):
                """
                :param func:
                :return:
                """

                if func.__doc__:
                    for line in func.__doc__.splitlines():
                        if line.strip():
                            return line.strip()
                    return ""

            # set description attribute for test framework to display
            doc = doc_gist(test_func)
            if doc:
                f.description = (test_name + ": ").ljust(10) + doc
            else:
                f.description = test_name.ljust(10)

            return f

        return func_wrapper

    @classmethod
    def do_test_rebase(cls):
        """
        :return: whether to run test in rebase mode, whether to not prompt for applying changes
        """
        val = os.getenv(ExeTestEnvVars.REBASE)
        if val is None:
            return False, False

        do_rebase = val != '0'
        return do_rebase, do_rebase and val != force_rebase

    def raise_exception(self, msg):
        raise Exception(msg)

    def run_test(self, exe_args, compare_spec,
                 pre_cmd, env_vars, post_cmd,
                 test_name, parent_module_name,
                 verbose):

        do_rebase, rebase_with_prompt = self.do_test_rebase()
        if rebase_with_prompt:
            if not sys.stdout.isatty():
                self.raise_exception("cannot rebase unless confirmation "
                                     "prompt is displayed in terminal. "
                                     "Make sure you are using --capture=no pytest option")

        with working_dir(self.test_root):

            files_to_compare = self.get_files_to_compare(test_name, compare_spec)

            if compare_spec and not files_to_compare:
                self.raise_exception(f"No reference output files for {compare_spec}")

            created_dirs = []
            for ref_file, new_file in files_to_compare:

                if os.path.isdir(ref_file):
                    file_dir = new_file
                else:
                    file_dir = os.path.dirname(new_file)

                if file_dir and not os.path.exists(file_dir):
                    topmost_created_dir = file_dir
                    while True:
                        parent_dir = os.path.split(topmost_created_dir)[0]
                        if not parent_dir or parent_dir == self.TMP_OUTPUT_DIR or os.path.exists(parent_dir):
                            break
                        else:
                            topmost_created_dir = parent_dir

                    os.makedirs(file_dir, exist_ok=True)
                    created_dirs.append(topmost_created_dir)

            tmp_output_dir = self.get_output_dir(test_name)
            run_from_dir = os.path.join(self.test_root, tmp_output_dir) \
                if self.run_from_out_dir else self.test_root

        if not self._compare_only:
            self.clear_dir(tmp_output_dir, recreate=True)

        with working_dir(run_from_dir):
            try:
                misc_utils.exec_cmdline(self.exe_path,
                                        exe_args,
                                        pre_cmd=pre_cmd,
                                        env_vars=env_vars,
                                        post_cmd=post_cmd,
                                        log_save_path=self.log_output_path,
                                        norun=self.norun or self._compare_only,
                                        verbose=verbose)
            except Exception as exc:
                if self.exception_handler:
                    self.exception_handler(exc)
                else:
                    raise

        if self.norun:
            if self.norun == 'ctest':
                print()
                print('|'.join(
                    str(item) for item in ((parent_module_name.replace('.', '/') + '.py::test_' + test_name),
                                           run_from_dir,
                                           ' '.join(f'{key}={val}' for key, val in env_vars.items()),
                                           os.path.basename(self.exe_path),
                                           exe_args)))
            if not self._compare_only:
                # this will mark the test as skipped
                pytest.skip("no-run mode")

        with working_dir(self.test_root):

            if do_rebase:
                self.run_rebase_compare(files_to_compare,
                                        force_rebase=not rebase_with_prompt)
            else:
                self.run_compare(files_to_compare)

            if not self._keep_output_on_success:
                for file_dir in created_dirs:
                    rmdir(file_dir)

    def run_compare(self, files_to_compare):

        for _ref_file, new_file in files_to_compare:
            if not os.path.exists(new_file):
                self.raise_exception(f"Missing output file: {new_file}")

        for ref_file, new_file in files_to_compare:
            has_ref = os.path.exists(ref_file)
            has_new = os.path.exists(new_file)
            if not has_ref and not has_new:
                self.raise_exception(f"Missing both ref and new files:\n{ref_file}\n{new_file}")
            else:
                if not has_ref:
                    self.raise_exception(f"Missing reference file: {ref_file} - "
                                         f"you can rebase with --rebase option")
                if not has_new:
                    self.raise_exception(f"Missing output file: {new_file}")

        for ref_file, new_file in files_to_compare:
            self.compare_equal(ref_file, new_file)

    def run_rebase_compare(self, files_to_compare, force_rebase=False):

        failed_rebase_msg = ''
        incomplete_rebase = False

        for ref_file, new_file in files_to_compare:

            if not os.path.exists(new_file) and os.path.exists(ref_file):
                print(f'{new_file} does not exist')
                if force_rebase or 'Y' == input("Remove it from reference output? (Y/[n]").strip():
                    if os.path.isdir(ref_file):
                        shutil.rmtree(ref_file)
                    else:
                        os.remove(ref_file)
                    print(f'removed {new_file} from reference output')
                continue

            if not os.path.exists(ref_file):
                os.makedirs(os.path.dirname(ref_file), exist_ok=True)
            elif self.compare_equal(ref_file, new_file, throw=False):
                continue

            print(f"rebasing test baseline: {new_file} -> {ref_file}")
            try:
                if force_rebase or 'Y' == input("Are you sure? (Y/[n]) ").strip():
                    try:
                        if os.path.isdir(new_file):
                            shutil.copytree(new_file, ref_file)
                            #shutil.rmtree(new_file)
                        else:
                            shutil.copy(new_file, ref_file)
                            #os.remove(new_file)
                    except PermissionError as err:
                        failed_rebase_msg += f'\ncp {new_file} {ref_file}'
                        print('rebase failed', str(err))
                    else:
                        print('rebase successful')
                else:
                    incomplete_rebase = True
            except KeyboardInterrupt:
                incomplete_rebase = True
                break

        if incomplete_rebase:
            raise unittest.SkipTest('incomplete rebase')

        if failed_rebase_msg:
            self.raise_exception(f'failed rebasing tests: {failed_rebase_msg}')

    def get_file_comparator(self, filepath):
        filename = os.path.basename(filepath)
        if filename in self.comparators:
            return self.comparators[filename]

        file_ext = filename.rsplit('.', 1)[-1]
        if file_ext in self.comparators:
            return self.comparators[file_ext]
        # default file comparator
        return FileComparator(max_diff_in_log=self._num_lines_diff)

    def compare_equal(self, ref_file, new_file, throw=True):

        compare_functors = self.get_file_comparator(ref_file)

        if compare_functors is None:
            # do not compare that file
            print(f"ignoring file: {ref_file}")
            return True

        max_len = max(len(ref_file), len(new_file)) + 10
        fmtd_file1 = ref_file.rjust(max_len)
        fmtd_file2 = new_file.rjust(max_len)
        files_info = f'\n{fmtd_file1}\n{fmtd_file2}'

        if not isinstance(compare_functors, (tuple, list)):
            compare_functors = [compare_functors]

        if self.verbose:
            print()

        for compare_functor in compare_functors:
            comparison_description = compare_functor.description()
            if comparison_description:
                comparison_description = f' ({comparison_description})'
            if compare_functor(ref_file, new_file):
                if self.verbose:
                    print(f'files match{comparison_description}: {files_info}')
            else:
                error_msg = f'files differ{comparison_description}: {files_info}'
                if self.verbose:
                    print(error_msg)
                if throw:
                    self.raise_exception(error_msg)
                else:
                    return False
        return True

    def make_dir_path(self, dir_stem, test_name, nested_dir):
        if self.test_name_as_dir:
            if nested_dir:
                return os.path.join(test_name, dir_stem)
            else:
                return os.path.join(dir_stem, test_name)
        else:
            return dir_stem

    def get_files_to_compare(self, test_name, compare_spec=None):
        """
        :param test_name: name of the test, used to deduce the ref dir path.
        :param compare_spec: specifies output files to compare; either:
                - a path to a file
                - a path to a directory (in which case all its contents are compared)
                - a list of those.
            if left to its None default, compares everything in ref_dir
        :return: a list of pairs of filepaths to compare: [(ref_file_path, new_file_path), ...]
        """

        ref_dir = self.get_ref_dir(test_name)

        if compare_spec is None:
            compare_spec = ref_dir

            if not os.path.exists(compare_spec):
                out_dir = self.get_output_dir(test_name)
                return [(ref_dir, out_dir)]

        elif not compare_spec:
            return []

        if isinstance(compare_spec, str):
            # single reference file
            compare_spec = [compare_spec]

        files_to_compare = []

        for ref_path in compare_spec:

            if isinstance(ref_path, tuple):
                file1, file2 = ref_path
                out_dir = self.get_output_dir(test_name)
                files_to_compare.append((os.path.join(ref_dir, file1),
                                         os.path.join(out_dir, file2)))
                continue
            elif isinstance(compare_spec, dict):
                new_path = compare_spec[ref_path]
            else:
                new_path = self.get_output_dir(test_name)

            if os.path.isdir(ref_path):
                # add all files under ref directory
                ref_path = os.path.normpath(ref_path)
                new_path = os.path.normpath(new_path)

                for dirpath, dirnames, filenames in os.walk(ref_path):
                    tmp_path = dirpath.replace(ref_path, new_path, 1)

                    for filename in filenames:
                        files_to_compare.append((os.path.join(dirpath, filename),
                                                 os.path.join(tmp_path, filename)))

            else:
                if not os.path.exists(ref_path):
                    ref_path = os.path.join(ref_dir, ref_path)

                # self.raise_exception(ref_path.replace(ref_dir, new_path))
                files_to_compare.append((ref_path, ref_path.replace(ref_dir, new_path)))

        return files_to_compare

    def clear_dir(self, dir_path, recreate=False):
        with working_dir(self.test_root):
            output_dir = dir_path
            if recreate:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
            else:
                # the only temporary files left behind should be
                # files exhibiting differences to reference directory.
                # Keep those around for investigation.
                for dirpath, dirnames, filenames in os.walk(output_dir):
                    if not filenames and not dirnames:
                        shutil.rmtree(dirpath)

            if recreate:
                os.makedirs(output_dir, exist_ok=True)
