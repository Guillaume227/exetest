# exetest
A library for conveniently writing and managing tests that compare the output of some executable to a golden copy.  


Supported features:
- standard python test discovery and execution with `pytest` / `nosetests`
- streamlined test rebasing when the golden copy changes


### Show me how it's done
Assuming you have an executable called `myexe`

the idea is to create a decorator by instanciating exetest.ExeTestCaseDecorator:
```
myexe_testcase = exetest.ExeTestCaseDecorator(
    exe=<path_to_exe>,
    test_root=<path-to-your-tests-directory>)

```

We then use that instance to decorate concise test cases:  
```
@myexe_testcase()
def test_case_simplest():
    """
    add test description as a docstring
    """
    pass # Note: A test case doesn't require any code.
         #       All the magic happens in the decorator.
```
The above snippet will run `myexe` with no arguments from a default directory called `out_dir` located under `test_root` specified above.
All the outputs produced by that run of the executable are then compared against reference files.
Those reference files are expected to be found in a directory `test_root/ref_dir`.

Note:
- Both `out_dir` and `ref_dir` can be customized.
- By default the files produced must match exactly the reference files. 

```
@myexe_testcase(
        exe_args=(arg1, arg2) # specify myexe arguments needed by that run if any
        ref_dir=<path-to-ref-dir>,
        out_dir=<path-to-out-dir>,
        compare_spec="*.csv", # specify which files are to be compared                
)

def test_case_advanced():
    """
    add test description here
    """
    pass
```



