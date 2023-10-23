from functools import wraps


def expects_exception(exception_type=Exception,
                      expected_message=None,
                      expected_message_validator=None):
    """
    A decorator to test that a function throws an exception.

    sample usage:
        @except_exception()
        def test_my_func():
            ...
            # some code here that should raise Exception
            ...

    the test will pass if calling test_my_func raises an Exception, otherwise it will raise.
    In addition, the exception message content can be asserted.

    :param exception_type: exception type that a call to the method should raise.
    :param expected_message: excepted Exception message
    :param expected_message_validator: a predicate that takes the exception message and returns True if it is valid.
    :return: decorated function
    """
    def decorator(test_func):

        @wraps(test_func)
        def wrapped_func(*args, **kwargs):
            try:
                ret = test_func(*args, **kwargs)
            except exception_type as exc:
                message = exc.args[0]
                if expected_message is not None:
                    assert(expected_message == message), \
                        "exception message is not as expected"

                if expected_message_validator is not None:
                    assert(expected_message_validator(message)), \
                        "exception message is not as expected"

                return

            raise Exception('Expected exception was not thrown. '
                            f'returned value: "{ret}"')
        return wrapped_func

    return decorator
