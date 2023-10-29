from exetest import ExeTestCaseDecorator, expects_exception
import os

parent_dir = os.path.dirname(__file__)
path_to_exe = os.path.join(parent_dir, 'myexe')
print('path_to_exe')
# Create a test case decorator for the executable we want to test:


def myexe_testcase(comparators=None, **kwargs):

    dec = ExeTestCaseDecorator(exe=path_to_exe,
                               test_root=parent_dir,
                               nested_ref_dir=False,
                               comparators=comparators)
    return dec(**kwargs)


@myexe_testcase()
def test_a():
    """
    basic scenario case
    """
    pass

'''
@myexe_testcase(exe_args='--log myexe.log',
                compare_spec="*.txt")
def test_ignore_log():
    pass


@expects_exception()
@myexe_testcase(exe_args='--log myexe.log')
def test_not_ignored_log():
    pass

'''


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


@myexe_testcase(comparators={'myexe_output.txt': None})
def test_bb():
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
