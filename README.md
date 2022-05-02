# exetest
A library for conveniently writing and managing tests that compare the output of some executable to a golden copy.  


Supported features:
- standard python test discovery and execution with `pytest` / `nosetests`
- streamlined test rebasing when the golden copy changes
- 

### Show me how it's done
Assuming you have an executable called `some_exe`
```
@SomeExeTestDecorator(
    ref_dir='golden_copy/dir',
    exe_args=(arg1, arg2)
)
def test_case1():
    pass
```


