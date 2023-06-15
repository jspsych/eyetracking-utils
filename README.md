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
## Pipeline Guide
This is how to create TensorFlow datasets using the eyetracking-utils package.

We start from a directory of jpg images uploaded from OSF. The images are the last frame from each webm video recorded during the experiment.  

First, make an empty directory where the TFRecord files will be stored. 
```python
!mkdir tfrecords
```

Then, run a function from the tfrecords_processing file, such as:

```python
extract_meshes_to_tfrecords(in_path = path/to/jpg/directory, out_path = path/to/empty/directory)
```

The functions in the tfrecrods_processing file will output a directory of TFRecord files. There will be one file per subject containing all their data. 

After that, you can create a TensorFlow dataset from the directory of TFRecord files. These functions come from the dataset_utils file.

```python
train_data, validation_data, test_data = util.process_tfr_to_tfds(directory_path = path/to/tfrecords, process = util.parse_tfr_element_mediapipe)
```

This line will create 3 TensorFlow datasets: training, validation, and testing data. You must use a processing function that matches the tfrecords_processing function you chose earlier. 

Lastly, before passing a TensorFlow dataset to a model, you must batch the dataset. 

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
