### Introduction
The provided scripts provide a tested mechanism to break captcha using open-cv image processing and cnn models.

### Before you get started

To run these scripts, you need the following installed:

1. Python 2.7
2. OpenCV 3 w/ Python extensions

3. The python libraries listed in requirements.txt
 - Try running "pip install -r requirements.txt"

### Step 1: Extract single letters from CAPTCHA images
Step 1: Add the path for the input images folder and the output folder to store the letters in the script

default_folders:
input-images - for storing input captcha images
extracted_letter_images - for storing output letters
 
Step2:
Run:
`python extract_single_letters_from_captchas.py`

The results will be stored in the output folder.


### Step 2: Train the neural network to recognize single letters

Step 1: Add the path to the output images folder as set in the previous part in the script

Step2:
Run:
`python train_model.py`

This will write out "captcha_model.hdf5" and "model_labels.dat"


### Step 3: Use the model to solve CAPTCHAs!
Step 1: Add the path of folder for the test images

Step 2:
Run: 
`python solve_captchas_with_model.py`

The current CNN model provided in the repository gives about 75% accuracy.....To test the model run `solve_captchas_with_model.py`
