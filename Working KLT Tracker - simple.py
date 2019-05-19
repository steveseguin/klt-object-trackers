import numpy as np
import cv2
import itertools

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
GObject.threads_init()
Gst.init(None)

# some constants and default parameters
lk_params = dict(winSize=(25,25),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)) 

subpix_params = dict(zeroZone=(-1,-1),winSize=(10,10),
                     criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

feature_params = dict(maxCorners=500,qualityLevel=0.03,minDistance=5)


class LKTracker(object):
	"""    Class for Lucas-Kanade tracking with 
		pyramidal optical flow."""

	def __init__(self):
		"""    Initialize with a list of image names. """
		self.features = []
		self.tracks = []
		self.bb = [10,10,20,20]
		cv2.namedWindow("image")
		self.somecallback = cv2.setMouseCallback("image", self.click_and_crop) ## Allow the user to click and select a ROI (small bug at times)

	def step(self,image):
		"""    Step to another frame. If no argument is 
			given, step to the next frame. """

		self.gray = image

	def detect_points(self):
		"""    Detect 'good features to track' (corners) in the current frame
			using sub-pixel accuracy. """

	

		feature_mask = np.zeros_like(self.gray) ## Create a mask so we only look for template features in the ROI
		
		feature_mask[max(0,self.bb[1]):min(360,self.bb[1] + self.bb[3]),max(0,self.bb[0]):min(640,self.bb[0] + self.bb[2])] = 255

		# search for good points
		features = cv2.goodFeaturesToTrack(self.gray, mask = feature_mask, **feature_params)
		# refine the corner locations
		cv2.cornerSubPix(self.gray,features, **subpix_params)

		self.features = features

		self.tracks = [[p] for p in features.reshape((-1,2))]

		self.prev_gray = self.gray

	def track_points(self, image):
		"""    Track the detected features. """

		if self.features != []:
			self.step(image) # move to the next frame
			
			# reshape to fit input format
			tmp = np.float32(self.features).reshape(-1, 1, 2)
			
			# calculate optical flow
			features,status,track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray,self.gray,tmp,None,**lk_params)

			# remove points lost
			self.features = [p for (st,p) in zip(status,features) if st]
			
			# clean tracks from lost points
			features = np.array(features).reshape((-1,2))
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
		#print np.shape(self.gray)
		#cv2.rectangle(self.gray, (self.bb[0], self.bb[1]),  (self.bb[0] + self.bb[2], self.bb[1] + self.bb[3]))

		# draw points as green circles
		for point in self.features:
			cv2.circle(self.gray,(int(point[0][0]),int(point[0][1])),3,(255),-1)
			
		cv2.imshow('image',self.gray)
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

	def click_and_crop(self, event, xx, yy, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.bb[0],self.bb[1] = max(0,int(xx-self.bb[2]/2)),max(0,int(yy-self.bb[3]/2))
			self.detect_points()
		
		

global lkt
lkt = LKTracker()
global image
global counter
counter=0
	
def on_new_buffer(appsink2):
	global image
	global lkt
	global counter
	
	w=640*2
	h=480
	size=w*h
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	print buf.get_size(),size
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	image=stream[0:size*2:2].reshape(h,w/2) # create the y channel same size as the image

	if counter==0:
			lkt.step(image)
			lkt.detect_points()
			counter+=1
	elif counter==50:  ## We update the reference features every 8 frames, in hopes that this allows the system to evolve as the object changes
		lkt.detect_points()
		counter=1
	else:
		counter+=1
	lkt.track_points(image)
	lkt.draw()
	return False
	
	
CLI2="ksvideosrc device-index=2 ! video/x-raw,format=YUY2,width=640,height=480,framerate=60/1 ! queue leaky=downstream max-size-buffers=1 ! appsink name=sink" ## 60 FPS
pipline2=Gst.parse_launch(CLI2)
appsink2=pipline2.get_by_name("sink")
appsink2.set_property("max-buffers",20) # prevent the app to consume huge part of memory
appsink2.set_property('emit-signals',True) #tell sink to emit signals
appsink2.set_property('sync',False) #no sync to make decoding as fast as possible
appsink2.set_property('drop', True) ##  qos also
appsink2.set_property('async', True)
appsink2.connect('new-sample', on_new_buffer) #connect signal to callable func
pipline2.set_state(Gst.State.PLAYING)
bus2 = pipline2.get_bus();
msg2 = bus2.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

