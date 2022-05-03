from exetest import ExeTestCaseDecorator, expects_exception
import os

parent_dir = os.path.dirname(__file__)
path_to_exe = os.path.join(parent_dir, 'myexe')

# Create a test case decorator for the executable we want to test:
myexe_testcase = ExeTestCaseDecorator(exe=path_to_exe,
                                      test_root=parent_dir,
                                      nested_ref_dir=True)


@myexe_testcase()
def test_d():
    """
    basic scenario case
    """
    pass


def validate_message(message):
    return 'files differ' in message


@expects_exception(expected_message_validator=validate_message)
@myexe_testcase()
def test_e():
    """
    this test is setup to fail with a difference
    in output compared to ref
    """
    pass


@myexe_testcase(exe_args='-N 4')
def test_f():
    """
    passing arguments to the exe to test
    """
    pass
