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

Note: Changing from branch to branch (i.e. main to dev branch) will not 
update the module in your notebook environment, you'll have to do a complete
wipe and restart of the environment in order for changes in the code to 
be reflected, but once done, sticking on that one branch and updating will
still work.

To import various files, just use:
```python
import et_util.filename as something 
```
And access the functions with:
```python
something.function()
```
## Pipeline Guide
This is how to create TensorFlow datasets using the eyetracking-utils package.

We start from a directory of jpg images uploaded from OSF. The images are the last frame from each webm video recorded during the experiment.  

First, make an empty directory where the TFRecord files will be stored. Then, run a function from the `tfrecords_processing.py` file, such as:

```python
extract_meshes_to_tfrecords(in_path = path/to/jpg/directory, out_path = path/to/empty/directory)
```

This creates a directory of TFRecord files with one file per subject. 

Using functions from the `dataset_utils.py` file, you can create TensorFlow datasets from the directory of TFRecord files:

```python
train_data, validation_data, test_data = process_tfr_to_tfds(directory_path = path/to/tfrecords, process = parse_tfr_element_mediapipe)
```

These functions require a process argument, which is a function that parses the TFRecords. The process function must match the `tfrecords_processing.py` function you chose earlier. You now have 3 TensorFlow datasets: training, validation, and testing data. 

Before passing a TensorFlow dataset to a model, you must batch the dataset. 

```python
train_data.batch(batch_size)
```


## Versioning
Until we release our v1 of this project, we will use a separate 
versioning system for this. On development branches, when pushing 
code, increment the last digit to ensure the code changes will update. 
Example: `version = "0.1.0"` -> `version = "0.1.1"`

Before you merge changes to the main branch, make sure you update the 
second digit in order to properly update the package for all. 
Example: `version = "0.1.12"` -> `version = "0.2.0"`
