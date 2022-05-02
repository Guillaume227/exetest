from exetest import expects_exception
import os
import pytest


def func_that_raises(exception_type=Exception, exception_arg=''):
    raise exception_type(exception_arg)


def test_expected_exception():
    expects_exception()(func_that_raises)()


def test_different_exception():
    with pytest.raises(Exception):
        # if we filter for ArithmeticError only, the below will still raise
        expects_exception(exception_type=ArithmeticError)(func_that_raises)()


def test_excepted_message():

    expects_exception(exception_type=ArithmeticError,
                      expected_message='expected message')(func_that_raises)(
        exception_type=ArithmeticError,
        exception_arg='expected message')


def test_different_message():

    with pytest.raises(Exception):
        # if we filter for ArithmeticError, the below will still raise
        expects_exception(exception_type=ArithmeticError,
                          expected_message='expected message')(func_that_raises)(
            exception_type=ArithmeticError,
            exception_arg='different message')

