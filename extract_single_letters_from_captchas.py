import os
import os.path
import cv2
import glob
import imutils
from copy import copy
from helpers import diff_pixels
CAPTCHA_IMAGE_FOLDER = "../../Captcha_1000"
OUTPUT_FOLDER = "extracted_letter_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file, cv2.cv2.IMREAD_GRAYSCALE)

    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)[1]

    # Apply denoising and bilateral filter to the black and white image
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

    # skip the image if 5 valid characters are not found
    if flag == 1:
        continue
    
    # sort the letter coordinated in increasing order of x for maintaing the correct order of the string
    letter_positions.sort(key= lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_positions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        #offset of two pixels around each contour
        offset = 2

        # adjust the width of the contour
        if (w < 15):    w = 13
        if(w > 30):     w = 28 

        # Extract the letter from the original image with a 2-pixel margin around the edge
        if x - offset >=0:
            letter_image = output_image[:, x - offset:x + w + offset]
        else:
            letter_image = output_image[:, x:x + w + offset]
            
        # Get the folder to save the image in based on the character
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # get the no of images stored already in that folder
        count = counts.get(letter_text, 1)

        # save the image as {count}.jpg in the folder
        p = os.path.join(save_path, "{}c.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
