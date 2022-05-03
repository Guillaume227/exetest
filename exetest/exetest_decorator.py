import os
import os.path
from . import misc_utils
from .misc_utils import working_dir, rmdir
from .diff_utils import default_file_diff
from functools import wraps
import shutil
import sys
import unittest


class ExeTestCase:
    """
    A test case decorator for testing an executable outputs
    by comparing new output to reference output
    """

    USE_EXE_ENV_VAR = 'USE_TEST_EXE'
    REBASE_ENV_VAR = 'DO_TEST_REBASE'
    COMPARE_ONLY_ENV_VAR = 'EXETEST_COMPARE_ONLY'
    EXETEST_VERBOSE_ENV_VAR = 'EXETEST_VERBOSE'

    def __init__(self,
                 exe,
                 ref_dir,
                 out_dir,
                 test_root,  # =os.path.dirname(__file__),
                 exe_args=None,
                 run_from_out_dir=True,
                 test_name_as_dir=True,
                 comparators=None,
                 exception_handler=None,
                 env_vars=None,
                 pre_cmd=None,
                 exe_path_ref=None,
                 log_output_path=None):
        """

        :param exe:
        :param ref_dir: absolute path or relative to directory where test case is defined
        :param out_dir:
        :param test_root:
        :param exe_args: test executable arguments in string format - as would be passed to command line
        :param run_from_out_dir: whether the executable working directory should be the output directory
        :param test_name_as_dir: whether the test name should be used to infer the ref and output directories
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
                return val
            else:
                return exe

        rebase_exe = os.environ.get(self.REBASE_ENV_VAR)
        use_test_exe = os.environ.get(self.USE_EXE_ENV_VAR)

        # if USE_TEST_EXE has special target value, run against reference executable
        if rebase_exe is not None:
            assert use_test_exe is None, f"{self.REBASE_ENV_VAR} and {self.USE_EXE_ENV_VAR} are mutually exclusive"
            self.exe_path = get_test_exe(rebase_exe)

        elif use_test_exe is not None:
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

        self.verbose = self.EXETEST_VERBOSE_ENV_VAR in os.environ

    @staticmethod
    def get_test_subdir(test_name):
        return test_name

    def get_output_dir(self, test_name):
        output_dir = self.TMP_OUTPUT_DIR

        if self.test_name_as_dir:
            output_dir = os.path.join(output_dir, test_name)
        return output_dir

    def __call__(self, exe_args=None, compare_spec=None, pre_cmd=None,
                    env_vars=None, post_cmd=None, owners=None):
        """

        :param exe_args:
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

        self._compare_only = self.COMPARE_ONLY_ENV_VAR in os.environ

        def func_wrapper(test_func):
            """
            :param test_func:
            :return:
            """
            test_name = test_func.__name__.split("_", 1)[-1]

            @wraps(test_func)
            def f(*args, **kwargs):
                ret = test_func(*args, **kwargs)
                try:
                    self.run_test(exe_args,
                                  compare_spec,
                                  pre_cmd=pre_cmds,
                                  env_vars=all_env_vars,
                                  post_cmd=post_cmd,
                                  test_name=test_name,
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
        :return: whether to run test in rebase mode
        """
        return os.getenv(cls.REBASE_ENV_VAR) is not None

    def run_test(self, exe_args, compare_spec, pre_cmd, env_vars, post_cmd, test_name, verbose):

        if self.do_test_rebase():
            if not sys.stdout.isatty():
                raise Exception("cannot rebase unless confirmation prompt is displayed in terminal"
                                "make sure you are using --nocapture option")

        with working_dir(self.test_root):

            files_to_compare = self.get_files_to_compare(compare_spec, test_name)

            if compare_spec and not files_to_compare:
                raise Exception(f"No reference output files for {compare_spec}")

            created_dirs = []
            for ref_file, new_file, in files_to_compare:

                if os.path.isdir(ref_file):
                    file_dir = new_file
                else:
                    file_dir = os.path.dirname(new_file)

                if file_dir and not os.path.exists(file_dir):
                    os.makedirs(file_dir, exist_ok=True)
                    created_dirs.append(file_dir)

            if not self._compare_only:

                tmp_output_dir = self.get_output_dir(test_name)
                self.clear_dir(tmp_output_dir, recreate=True)

                run_from_dir = os.path.join(self.test_root, tmp_output_dir) \
                    if self.run_from_out_dir else self.test_root

                with working_dir(run_from_dir):
                    try:
                        misc_utils.exec_cmdline(self.exe_path, exe_args,
                                                pre_cmd=pre_cmd,
                                                env_vars=env_vars,
                                                post_cmd=post_cmd,
                                                log_save_path=self.log_output_path,
                                                verbose=verbose)
                    except Exception as exc:
                        if self.exception_handler:
                            self.exception_handler(exc)
                        else:
                            raise

            with working_dir(self.test_root):

                for _ref_file, new_file in files_to_compare:
                    if not os.path.exists(new_file):
                        raise Exception(f"Missing output file: {new_file}")

                if self.do_test_rebase():
                    self.run_rebase_compare(files_to_compare)
                else:
                    self.run_compare(files_to_compare)

                for file_dir in created_dirs:
                    rmdir(file_dir)

    def run_compare(self, files_to_compare):

        for ref_file, _new_file in files_to_compare:
            if not os.path.exists(ref_file):
                raise Exception(f"Missing reference file: {ref_file} - you can rebase by using {self.REBASE_ENV_VAR}= environment variable")

        for ref_file, new_file in files_to_compare:
            self.diff_files(ref_file, new_file)

    def run_rebase_compare(self, files_to_compare):

        failed_rebase_msg = ''

        for ref_file, new_file in files_to_compare:
            if not os.path.exists(ref_file):
                os.makedirs(os.path.dirname(ref_file), exist_ok=True)
            elif self.diff_files(ref_file, new_file, throw=False):
                continue

            print(f"rebasing test - about to update gold copy: {new_file} -> {ref_file}")

            if 'Y' == input("Are you sure? (Y/n) ").strip():
                try:
                    if os.path.isdir(new_file):
                        shutil.copytree(new_file, ref_file)
                        shutil.rmtree(new_file)
                    else:
                        shutil.copy(new_file, ref_file)
                        os.remove(new_file)
                except PermissionError as err:
                    failed_rebase_msg += f'\ncp {new_file} {ref_file}'
                    print('rebase failed', str(err))
            else:
                print("aborting rebase for", ref_file)

        if failed_rebase_msg:
            raise Exception(f'failed rebasing tests: {failed_rebase_msg}')

    def diff_files(self, ref_file, new_file, throw=True):

        file_ext = ref_file.rsplit('.', 1)[0]
        comparator_func = self.comparators.get(file_ext, default_file_diff)

        max_len = max(len(ref_file), len(new_file)) + 10
        fmtd_file1 = ref_file.rjust(max_len)
        fmtd_file2 = new_file.rjust(max_len)
        files_info = f'\n{fmtd_file1}\n{fmtd_file2}'

        if comparator_func(ref_file, new_file):
            print(f"matching outputs:{files_info}")
            return True
        elif throw:
            error_msg = f'files differ:{files_info}'
            raise Exception(error_msg)
        else:
            return False

    def infer_new_from_ref(self, ref_file_path, test_subdir):
        if ref_file_path.startswith(self.REF_OUTPUT_DIR):
            relative_dir = ref_file_path.split(self.REF_OUTPUT_DIR)[-1]
            return os.path.join(self.TMP_OUTPUT_DIR, relative_dir[1:])
        elif not os.path.isabs(ref_file_path):
            return os.path.join(self.TMP_OUTPUT_DIR, test_subdir, ref_file_path)
        elif os.path.isdir(ref_file_path):
            return os.path.join(self.TMP_OUTPUT_DIR, test_subdir)
        else:
            return os.path.join(self.TMP_OUTPUT_DIR, test_subdir, os.path.basename(ref_file_path))

    def get_files_to_compare(self, compare_spec, test_name):
        """
        :param compare_spec: a specification of which output files to compare
        :param test_name: name of the test
        :return: a list of pairs of filepaths to compare: [(ref_file_path, new_file_path), ...]
        """
        if compare_spec is None:
            if self.test_name_as_dir:
                compare_spec = os.path.join(self.REF_OUTPUT_DIR, test_name)
            else:
                compare_spec = self.REF_OUTPUT_DIR

            if not os.path.exists(compare_spec):
                if self.test_name_as_dir:
                    return (compare_spec, os.path.join(self.TMP_OUTPUT_DIR, test_name)),
                else:
                    return (compare_spec, self.TMP_OUTPUT_DIR),

        elif not compare_spec:
            return []

        if isinstance(compare_spec, str):
            # single reference file
            compare_spec = [compare_spec]

        files_to_compare = []

        for ref_path in compare_spec:

            if isinstance(compare_spec, dict):
                new_path = compare_spec[ref_path]
            else:
                new_path = self.infer_new_from_ref(ref_path, test_name)

            if os.path.isdir(ref_path):
                # add all files under ref directory
                ref_path = os.path.normpath(ref_path)
                new_path = os.path.normpath(new_path)

                for dirpath, dirnames, filenames in os.walk(ref_path):
                    if self.run_from_out_dir:
                        tmp_path = new_path
                    else:
                        tmp_path = dirpath.replace(ref_path, new_path, 1)

                    for filename in filenames:
                        files_to_compare.append((os.path.join(dirpath, filename), os.path.join(tmp_path, filename)))
            else:

                if not os.path.exists(ref_path):
                    if self.test_name_as_dir:
                        test_subdir = self.get_test_subdir(test_name)
                        if not os.path.exists(os.path.join(self.REF_OUTPUT_DIR, test_subdir)):
                            raise Exception(f'missing test subdirectory: {test_subdir} in {self.REF_OUTPUT_DIR}')
                    else:
                        test_subdir = ''

                    ref_path = os.path.join(self.REF_OUTPUT_DIR, test_subdir, ref_path)

                files_to_compare.append((ref_path, new_path))

        return files_to_compare

    def clear_tmp_dir(self, recreate=False):
        self.clear_dir(self.TMP_OUTPUT_DIR, recreate=recreate)

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
