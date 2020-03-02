
# coding: utf-8

# In[25]:


from __future__ import print_function
import cv2
import numpy as np
import glob 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(img1, img2):
  #this converts all the images into gray scale
  img1Gr = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2Gr = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, d1 = orb.detectAndCompute(img1Gr, None)
  keypoints2, d2 = orb.detectAndCompute(img2Gr, None)
  # feature points are matched between two images(planes)
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(d1, d2, None)
  # sorting the scores of matches
  matches.sort(key=lambda x: x.distance, reverse=False)
  # dropping poor matching features
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  # selecting the best matching features
  imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  # Extract the location of those matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  height, width, channels = img2.shape
  img1Reg = cv2.warpPerspective(img1, h, (width, height))
  return img1Reg, h
 
 
if __name__ == '__main__':
  # Reading orginal image
  ref = "C:/Users/SRI HARI/Desktop/augmented_new/Original/original.jpg"
  imReference = cv2.imread(ref, cv2.IMREAD_COLOR)
  inpath = "C:/Users/SRI HARI/Desktop/augmented_new/Test_images/*.jpg"
  outpath = "C:/Users/SRI HARI//Desktop/augmented_new/out/"
  files = glob.glob(inpath)
  # Read image to be aligned
  cnt = 0
  #print("check")
  for imFilename in files :
      cnt+=1
      #print("check1");  
      im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
      #print("check alignment!") 
      imReg, h = alignImages(im, imReference)
      # writeing aligned imgs to destination 
      outFile = str(outpath) + "out" + str(cnt) + ".jpg"
      cv2.imwrite(outFile, imReg)
  print("SUCCESSfully aligned! :-)");

