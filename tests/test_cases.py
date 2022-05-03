from exetest import ExeTestCase, expects_exception
import os

parent_dir = os.path.dirname(__file__)
path_to_exe = os.path.join(parent_dir, 'myexe')

myexe_testcase = ExeTestCase(exe=path_to_exe,
                             ref_dir='ref_dir',
                             out_dir='out_dir',
                             test_root=parent_dir)


@myexe_testcase()
def test_a():
    """
    basic scenario case
    """
    pass


def validate_message(message):
    return 'files differ' in message


@expects_exception(expected_message_validator=validate_message)
@myexe_testcase()
def test_b():
    """
    this test is setup to fail with a difference
    in output compared to ref
    """
    pass


@myexe_testcase(exe_args='-N 4')
def test_c():
    """
    passing arguments to the exe to test
    """
    pass
