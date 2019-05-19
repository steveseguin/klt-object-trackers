from numpy import *
import cv2
import itertools
import serial


# some constants and default parameters
lk_params = dict(winSize=(15,15),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)) 

subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

feature_params = dict(maxCorners=1000,qualityLevel=0.02,minDistance=3)


class LKTracker(object):
	"""    Class for Lucas-Kanade tracking with 
		pyramidal optical flow."""

	def __init__(self,cap):
		"""    Initialize with a list of image names. """
		self.cap = cap
		self.bb = [int(1280/2-50), int(720/2-100), 80, 180]  ## ROI ; a rectangle of fixed shape -- this needs improvement -- Using a non-rigid mask based on the moving shpae's contour would be much better me thinks.
		r, self.image = cap.read()
		self.features = []
		self.old_feature = []
		self.tracks = []
		self.angle = 90

		try:
			self.ser = serial.Serial(7) # COM PORT FOR SERIAL (USB-based) SERVO CONTROLLER
			self.ser.baudrate = 9600
			self.gimbal()
		except serial.serialutil.SerialException:
			print "Servo Controller not found"
			self.ser = None
		cv2.namedWindow("image")
		self.somecallback = cv2.setMouseCallback("image", self.click_and_crop) ## Allow the user to click and select a ROI (small bug at times)

	def step(self):
		"""    Step to another frame. If no argument is 
			given, step to the next frame. """

		r, self.image = self.cap.read()

	def detect_points(self):
		"""    Detect 'good features to track' (corners) in the current frame
			using sub-pixel accuracy. """

		# create grayscale
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

		feature_mask = zeros_like(self.gray) ## Create a mask so we only look for template features in the ROI
		
		feature_mask[max(0,self.bb[1]):min(720,self.bb[1] + self.bb[3]),max(0,self.bb[0]):min(1280,self.bb[0] + self.bb[2])] = 255

		# search for good points
		features = cv2.goodFeaturesToTrack(self.gray, mask = feature_mask, **feature_params)
		# refine the corner locations
		cv2.cornerSubPix(self.gray,features, **subpix_params)

		self.features = features
		self.old_features = features

		self.tracks = [[p] for p in features.reshape((-1,2))]

		self.prev_gray = self.gray

	def track_points(self):
		"""    Track the detected features. """

		if self.features != []:
			self.step() # move to the next frame
			
			# load the image and create grayscale
			self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
			
			# reshape to fit input format
			tmp = float32(self.features).reshape(-1, 1, 2)
			
			# calculate optical flow
			features,status,track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,self.gray,tmp,None,**lk_params)

			# remove points lost
			self.features = [p for (st,p) in zip(status,features) if st]
			
			# clean tracks from lost points
			features = array(features).reshape((-1,2))
			for i,f in enumerate(features):
				self.tracks[i].append(f)
			ndx = [i for (i,st) in enumerate(status) if not st]
			ndx.reverse() #remove from back
			for i in ndx:
				self.tracks.pop(i)
			
			self.prev_gray = self.gray

	def draw(self):
		"""    Draw the current image with points using
			OpenCV's own drawing functions. 
			Press ant key to close window."""


		self.predict()
		cv2.rectangle(self.image, (self.bb[0], self.bb[1]),  (self.bb[0] + self.bb[2], self.bb[1] + self.bb[3]), color=(255, 0, 0))

		# draw points as green circles
		for point in self.features:
			cv2.circle(self.image,(int(point[0][0]),int(point[0][1])),3,(0,255,0),-1)
			
		self.old_features = self.features # save for next time

		cv2.imshow('image',self.image)
		cv2.waitKey(1)
		
	def predict(self):
		if len(self.features)==0:
			return
		dx = []
		dy = []
		for i in range(len(self.features)): # Only include features that are not extremely far off from the ROI zone -- this concept needs improvement
			if ((self.features[i][0][0]>=self.bb[0]-self.bb[2]) & (self.features[i][0][0]<=(self.bb[0]+self.bb[2]*2)) & (self.features[i][0][1]>=self.bb[1]-self.bb[3]) & (self.features[i][0][1]<=(self.bb[1]+self.bb[3]*2))): 
				dx.append( self.features[i][0][0])
				dy.append( self.features[i][0][1])
		if not dx or not dy:
			return
			
		cen_dx = round(sum(dx) / len(dx) - self.bb[2]/2)
		cen_dy = round(sum(dy) / len(dy) - self.bb[3]/2)

		self.bb = [int(cen_dx), int(cen_dy), self.bb[2], self.bb[3]]

		return

	def gimbal(self):
		if self.ser == None:
			return
		speed = 7. # speed; adjust as needed
		adjustment = (1280/2.0 - (self.bb[0]+self.bb[2]/2.0))/(1280)*speed
		if abs(adjustment)<0.05*speed:
			return
		self.angle = self.angle + adjustment 
		if self.angle>180:
			self.angle = 180
		elif self.angle < 0:
			self.angle = 0
			
		byteone=int(254*self.angle/180)
		#move to an absolute position in 8-bit mode (0x04 for the mode, 0 for the servo, 0-255 for the position (spread over two bytes))
		bud=chr(0xFF)+chr(0)+chr(byteone)
		self.ser.write(bud)
		return

	def click_and_crop(self, event, xx, yy, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.bb[0],self.bb[1] = max(0,int(xx-self.bb[2]/2)),max(0,int(yy-self.bb[3]/2))
			self.detect_points()
		
		
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
r, image = cap.read()

lkt = LKTracker(cap)

lkt.detect_points()
lkt.draw()

counter=0
while (cap.isOpened()):
	#if counter==100:  ## We update the reference features every 8 frames, in hopes that this allows the system to evolve as the object changes
	#lkt.detect_points()
	#	counter=0

	#else:
	#	counter+=1
	lkt.step()
	lkt.track_points()
	lkt.gimbal()
	lkt.draw()

