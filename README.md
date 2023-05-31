# eyetracking-utils
This package is a set of utilities for the eyetracking
project. To install, use this command:
```
!pip install git+https://{user}:{token}@github.com/jspsych/eyetracking-utils.git
```
Replace `{user}` with your Github user and `{token}` with a 
personal access token that has repo permissions. To access 
a specific branch, add `@{branch_name}` to the end of the package.

If you *ever* make changes to the code, make sure to increment the 
version in `pyproject.toml`. Otherwise, the changes will not update 
when you rerun the pip command.

To import various files, just use:
```python
import et_util.filename as something 
```
And access the functions with:
```python
something.function()
```

## Versioning
Until we release our v1 of this project, we will use a separate 
versioning system for this. On development branches, when pushing 
code, increment the last digit to ensure the code changes will update. 
Example: `version = "0.1.0"` -> `version = "0.1.1"`

Before you merge changes to the main branch, make sure you update the 
second digit in order to properly update the package for all. 
Example: `version = "0.1.12"` -> `version = "0.2.0"`