from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
from copy import copy
from helpers import diff_pixels
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

#Add the path of the folder of images to be tested
CAPTCHA_IMAGE_FOLDER = "/home/aditya/veratech/Captcha_2000"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
#captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

# loop over the image paths
counter = 0
for image_file in captcha_image_files:
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)[1]

    # apply denoising and bilateral filter to the black and white image
    filtered_image = cv2.fastNlMeansDenoising(thresh,None,10,7,21)
    output_image = cv2.bilateralFilter(filtered_image,7,110,110)

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours[0] if imutils.is_cv2() else contours[1]
    # list for storing the areas of all the contours found
    list1 = []	

	# finding area of each contour
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        area = w*h
        list1.append(area)

    # make a copy of list1 to find the index of the contours afterwards 
    list2 = copy(list1)

    #sorting the contours based on the area
    list1.sort()
    # array for storing the positions of the characters in the image
    letter_positions = []
    # count for no of valid contours (letters) found
    count = 0
    # index of the second largest contour in the list1 ( the largest contour contains the whole image)
    i = -2
    # flag for finding invalid training data
    flag = 0

    # the main control loop for finding the positions of the 5 characters. 
    # We iterate the list1 (sorted based on the area of contours) from the end
    # and check the validity of each contour 
    while count < 5:
        #find the index of the contour
        ind = list2.index(list1[i])

        #get the coordinates of the bounding rectangle of the contour
        x,y,w,h = cv2.boundingRect(contours[ind])

        #check the % of white pixels in the contour
        white_pixels =  float(float(cv2.countNonZero(thresh[:,x-2:x+w+2]))/float(((w+4)*35)))
        
        # if white pixels < 0.9 and the contour found does not overlap with previously found
        # contour, then contour is a valid letter and its coordinated are appended to letter_positions
        if white_pixels <= 0.90 and diff_pixels(x,letter_positions):
            letter_positions.append([x,y,w,h])
            count = count + 1
        
        i = i-1
        
        # if we have iterated all the contours without finding 5 valid letters
        if(abs(i) >= len(contours)):
            flag = 1
            break

    # sort the letter coordinated in increasing order of x for maintaing the correct order of the string
    letter_positions.sort(key= lambda x: x[0])

    predictions = []
    # loop over the letters
    output = cv2.merge([image] * 3)
    for i in range(0,len(letter_positions)):
        # Grab the coordinates of the letter in the image
        x,y,w,h = letter_positions[i]

        # an offset of two pixel around each contour
        offset = 2

        # adjust the width of the contour
        if (w < 15):
            w = 13
        if(w > 30):
            w = 28

        # Extract the letter from the original image with a 2-pixel margin around the edge
        if x - offset >= 0:
            letter_image = output_image[:, x - offset:x + w + offset]
        else:
            letter_image = output_image[:, x:x + w + 2*offset]            
        # cv2.imshow('a',letter_image)
        # cv2.waitKey()
        # Re-size the letter image to 20x20 pixels to match training data
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normals letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    string_name = image_file.split('/')[-1].split('.')[0]
    if captcha_text==string_name:
    	counter += 1
    print("CAPTCHA text is: {}".format(captcha_text))
print(counter)

