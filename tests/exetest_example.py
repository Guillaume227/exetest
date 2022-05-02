from exetest import exetest_decorator


def make_test_decorator():

    def decorator():
        return exetest_decorator.ExeTestDecorator()

    return decorator