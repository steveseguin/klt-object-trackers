# Python 2/3 compatibility
from __future__ import print_function
import sys, time
PY3 = sys.version_info[0] == 3


if PY3:
    xrange = range

import numpy as np
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
Gst.init(None)
Gst.debug_set_active(True)
Gst.debug_set_default_threshold(3)

eps = 1e-5

class MOSSE:
	def __init__(self, frame, rect):
		x1, y1, x2, y2 = rect
		self.pos = x,y = x1, y1
		self.size = w,h = x2-x1, y2-y1

		self.win = cv2.createHanningWindow((h, w), cv2.CV_32F)

		img = frame[y:y+h, x:x+w]
		img = self.preprocess(img)

		dct = cv2.dct(np.float32(img)/255.0)
		dct = dct[0:32, 0:18]

		idct = cv2.idct(dct)
		self.last_img = np.uint8(np.clip(idct/idct.max(), 0, 1)*255)

		self.H = dct

		self.update(frame)

	def update(self, frame, rate = 0.125):
		(x, y), (w, h) = self.pos, self.size

		img = frame[y:y+h, x:x+w]
		img = self.preprocess(img)
		dct = cv2.dct(np.float32(img)/255.0)
		dct = dct[0:32, 0:18]

		self.last_resp, (dx, dy), self.psr = self.correlate(dct)
		self.good = self.psr > 8.0
		if not self.good:
			return

		#self.pos = x+round(dx*w/18), y+round(dy*h/32)
		#x,y = self.pos

		img = frame[y:y+h, x:x+w]
		img = self.preprocess(img)
		dct = cv2.dct(np.float32(img)/255.0)
		dct = dct[0:32, 0:18]

		idct = cv2.idct(dct)
		self.last_img = np.uint8(np.clip(idct/idct.max(), 0, 1)*255)
		self.last_img[dy,dx]=255 ####
		#H = cv2.mulSpectrums(dct, dct, 0, conjB=True)
		self.H = self.H * (1.0-rate) + dct * rate

	@property
	def state_vis(self):

		resp = self.last_resp
		resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
		vis = np.hstack([self.last_img, resp])
		return vis

	def draw_state(self, vis):
		(x, y), (w, h) = self.pos, self.size
		x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
		cv2.rectangle(vis, (x1, y1), (x2, y2), (255))
		if self.good:
			cv2.circle(vis, (int(x), int(y)), 2, (255), -1)
		else:
			cv2.line(vis, (x1, y1), (x2, y2), (255))
			cv2.line(vis, (x2, y1), (x1, y2), (255))
		cv2.imshow("vis",vis)

	def preprocess(self, img):
		img = np.log(np.float32(img)+1.0)
		img = (img-img.mean()) / (img.std()+eps)
		return img*self.win

	def correlate(self, dct):
		C = cv2.absdiff(dct, self.H)#, 0, conjB=True)
		resp = cv2.idct(C)

		cv2.imshow("C",np.uint8(np.clip(C/C.max(), 0, 1)*255))

		h, w = resp.shape
		psr = 10
		return resp, (0, 0), psr




class SES():	
	trackers = []

global ses
ses = SES()
ses.counter=10

def on_new_buffer(appsink):
	global ses
	w=640
	h=480
	size=w*h*3 #2
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	ses.frame=np.array(stream[0:size:3].reshape(h,w)) # create the y channel same size as the image

	if ses.counter>0:
		rect = 300, 200, 400, 300
		if ses.counter==1:
			tracker = MOSSE(ses.frame, rect)
			ses.trackers.append(tracker)
		ses.counter-=1

	for tracker in ses.trackers:
		tracker.update(ses.frame)
	vis = ses.frame.copy()
	for tracker in ses.trackers:
		tracker.draw_state(vis)
	if len(ses.trackers) > 0:
		cv2.imshow('tracker state', ses.trackers[-1].state_vis)

	cv2.waitKey(1)
	return False
				
if __name__ == '__main__':
	
	CLI2="ksvideosrc device-index=0 ! video/x-raw,format=BGR,width=640,height=480,framerate=30/1 ! appsink name=sink" ## 60 FPS
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
	while True:
		time.sleep(10000)
	pipline2.set_state(Gst.State.NULL)
	