#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import shutil
import cv2
import random
import pytesseract
import matplotlib.pyplot as plt
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[49]:


def recognize_plate(img):
#     cv2.imshow("image", img)
#     cv2.waitKey(0)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     cv2.imshow("gray", gray)
#     cv2.waitKey(0)
    
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#     cv2.imshow("gray_resized", gray)
#     cv2.waitKey(0)
    
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
#     cv2.imshow("blur", blur)
#     cv2.waitKey(0)
    
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#     cv2.imshow("Otsu Threshold", thresh)
#     cv2.waitKey(0)
    
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
#         if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
#         cv2.imshow("roi", roi)
#         cv2.waitKey(0)
#         plt.imshow(roi)
#         plt.show()
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        roi = cv2.medianBlur(roi, 5)
#         print(roi)
#         plt.imshow(roi)
#         plt.show()
#         cv2.imshow('roi', roi)
        try:
            text = pytesseract.image_to_string(roi,
                config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            print(text)
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except: 
            text = None
    if plate_num != None:
        print("License Plate #: ", plate_num)
#     cv2.destroyAllWindows()
    return plate_num


# In[15]:


import pandas as pd 
dataset = pd.read_csv('C:/Users/hp/Desktop/IDFY/dataset.csv', names = ['Images','Plate'])


# In[16]:


dataset
dataset = dataset.sort_values('Images')
dataset


# In[12]:


# sources = ['C:/Users/hp/Desktop/IDFY/HDR/','C:/Users/hp/Desktop/IDFY/normal/']
target = 'C:/Users/hp/Desktop/IDFY/test_set/'
# for source in sources:
#     l = os.listdir(source)
#     for folder in l:
#         path = source + folder
#         images = os.listdir(path)
#         for img in images:
#             shutil.copy(path + '/' + img, target + "".join(folder+ img) )
        


# In[50]:


plates_esm = []
for img in os.listdir(target):
    print(img)
    image = cv2.imread(target + img)
    plt.imshow(image)
    plt.show()
    plates_esm.append(recognize_plate(image))


# In[57]:


print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy") 
print("--------------------", "\t", "-----------------------", "\t", "--------") 

def calculate_predicted_accuracy(actual_list, predicted_list):
    for actual_plate, predict_plate in zip(actual_list, predicted_list): 
        accuracy = '0'
        num_matches = 0
        if actual_plate == predict_plate: 
            accuracy = "100 %"
        else: 
            if len(actual_plate) == len(predict_plate): 
                for a, p in zip(actual_plate, predict_plate): 
                    if a == p:
                        num_matches += 1
                accuracy = str(round((num_matches / len(actual_plate))*100)) 
#                 accuracy = '45'
                accuracy += "%"
        inf['Actual License Plate'].append(actual_plate)
        inf['Predicted License Plate'].append(predict_plate)
        inf['Accuracy'].append(accuracy)
        print("  ", actual_plate, "\t\t\t", predict_plate, "\t\t  ", accuracy) 
        count.append(1)

count = []
inf = {'Actual License Plate':[], 'Predicted License Plate':[], 'Accuracy':[]}
calculate_predicted_accuracy(dataset.Plate, plates_esm)
print(len(count))


# In[64]:


inference = pd.DataFrame(inf)
inference.to_csv('Inference.csv', index = False)

