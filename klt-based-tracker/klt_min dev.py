import numpy as np
import cv2
import itertools

# --------------------------------------------  some constants and default parameters
lk_params = dict(winSize=(15,15),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)) 

subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

feature_params = dict(maxCorners=500,qualityLevel=0.5,minDistance=3)

# ----------------------------------
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# ------------- INIT --------------------
gray = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2GRAY)
tracks = []

def doubleMADsfromMedian(y,thresh=0.5):
    # warning: this function does not check for NAs
    # nor does it address issues when 
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y<=m])
    right_mad = np.median(abs_dev[y>=m])
    y_mad = np.zeros(len(y))
    y_mad[y < m] = left_mad
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh

while (cap.isOpened()):

	prev_gray = gray.copy()
	gray = cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2GRAY)
	features = cv2.goodFeaturesToTrack(gray, **feature_params)  # search for good points
	if features == []:
		continue
	cv2.cornerSubPix(gray, features, **subpix_params)  # refine the corner locations
	features_reshaped = np.float32(features).reshape(-1, 1, 2) # reshape to fit input format
	features,status,track_error = cv2.calcOpticalFlowPyrLK(prev_gray,gray,features_reshaped,None,**lk_params)
	features = [p for (st,p) in zip(status,features) if st] # remove points lost
	if features == []:
		continue
	features_reshaped = np.float32(features).reshape(-1, 1, 2) # reshape to fit input format
	features2, status2, track_error2 = cv2.calcOpticalFlowPyrLK(gray, prev_gray, features_reshaped, None, **lk_params)
	features2 = np.array(features2).reshape((-1,2))
	
	delta = features[0]-features2
	
	mask1 = doubleMADsfromMedian(delta[:,0])
	mask2 = doubleMADsfromMedian(delta[:,1])
	mask = mask1*mask2
	#print delta[mask]
	dist1=0
	dist2=0
	counter=0
	for i, point in enumerate(features):
		if mask[i]:
			dist1 += (point[0][0] - features2[i][0]) 
			dist2 += (point[0][1] - features2[i][1])
			counter+=1
			cv2.circle(gray,(int(point[0][0]),int(point[0][1])),3,(255),-1)
			cv2.circle(gray,(int(features2[i][0]),int(features2[i][1])),3,(255),-1)
			cv2.line(gray, (int(point[0][0]),int(point[0][1])), (int(features2[i][0]),int(features2[i][1])), (255))
	try:
		print dist1/counter, dist2/counter
	except:
		pass
	cv2.imshow("image",gray)
	cv2.waitKey(1)