import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import numpy as np
import zbar
import cv2
# create a reader
scanner = zbar.ImageScanner()
# configure the reader
scanner.parse_config('enable')

def on_new_buffer(appsink):
	w=1280
	h=720
	size=w*h*2
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	gray=255-stream[0:size:2].reshape(h,w) # create the y channel same size as the image
	
	raw=gray.tostring()
	cv2.imshow("image",gray)
	cv2.waitKey(1)
	# wrap image data
	image = zbar.Image(w, h, 'Y800', raw)

	# scan the image for barcodes
	scanner.scan(image)

	# extract results
	for symbol in image:
		# do something useful with results
		print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data

	
	return False


################
Gst.init(None)

CLI2="ksvideosrc device-index=2 ! video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! appsink name=sink" ## 60 FPS
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