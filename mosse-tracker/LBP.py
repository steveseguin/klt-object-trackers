import numpy as np
import cv2
from matplotlib import pyplot as plt


winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,360)
cap.set(5,30)
_, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

hist = hog.compute(img)*256

plt.hist(hist,256,[0,256]); plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()