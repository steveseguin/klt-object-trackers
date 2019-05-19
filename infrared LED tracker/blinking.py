import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import numpy as np
import cv2

global last
last = np.zeros((480,640)).astype("uint8")


def on_new_buffer(appsink):
	global last
	w=640
	h=480
	size=w*h*2
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	gray=np.array(stream[0:size:2].reshape(h,w)) # create the y channel same size as the image
	
	cv2.imshow("gray",gray)

	gray[np.where(gray<255)]=0
	#gray = cv2.GaussianBlur(gray,(5,5),0)
	cv2.imshow("gray2",gray)
	move = cv2.absdiff(gray,last) 
	
	deltasum = move.sum()
	if deltasum>0:
		y = range(0,move.shape[0])
		x = range(0,move.shape[1])
		(X,Y) = np.meshgrid(x,y)
		x_cord = float((X*move).sum()) / deltasum
		y_cord = float((Y*move).sum()) / deltasum
		print x_cord,y_cord
	
	last=gray
	cv2.imshow("move",move)
	cv2.waitKey(1)
	return False


################
Gst.init(None)

CLI2="ksvideosrc device-index=2 ! video/x-raw,format=YUY2,width=640,height=480,framerate=60/1 ! appsink name=sink" ## 60 FPS
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