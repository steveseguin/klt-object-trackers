import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

import argparse
import cv2
from numpy import empty, nan
import os
import sys
import time

import CMT
import numpy as np
import util

global ROI, im_prev, im_gray, frame, toc, tic, tl, br, w, h, size
ROI = False


CMT = CMT.CMT()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_false', help='Disable rotation estimation')
parser.add_argument('--quiet', dest='quiet', action='store_true', help='Do not show graphical output (Useful in combination with --output-dir ).')


args = parser.parse_args()

CMT.estimate_scale = args.estimate_scale
CMT.estimate_rotation = args.estimate_rotation


def on_new_buffer(appsink):
	global ROI, im_prev, im_gray, frame, toc, tic, tl, br
	print("asdf")
	w=1920
	h=1080
	size=w*h*3
	sample = appsink2.emit('pull-sample')
	buf=sample.get_buffer()
	print( buf.get_size(),size)
	data=buf.extract_dup(0,buf.get_size())
	stream=np.fromstring(data,np.uint8) #convert data form string to numpy array // THIS IS YUV , not BGR.  improvement here if changed on gstreamer side
	im_gray=np.array(stream[0:size:3].reshape(h,w)) # create the y channel same size as the image
	
	if ROI == False:
		cv2.imshow('Preview', im_gray)
		k = cv2.waitKey(1)
		if k == -1:
			return False
			
		(tl, br) = util.get_rect(im_gray)

		print( 'using', tl, br, 'as init bb')

		CMT.initialise(im_gray, tl, br)

		frame = 1
		ROI = True
	else:
		tic = time.time()
		CMT.process_frame(im_gray)
		toc = time.time()

		# Display results

		# Draw updated estimate
		if CMT.has_result:

			cv2.line(im_gray, CMT.tl, CMT.tr, (255), 4)
			cv2.line(im_gray, CMT.tr, CMT.br, (255), 4)
			cv2.line(im_gray, CMT.br, CMT.bl, (255), 4)
			cv2.line(im_gray, CMT.bl, CMT.tl, (255), 4)

		util.draw_keypoints(CMT.tracked_keypoints, im_gray, (255))
		# this is from simplescale
		util.draw_keypoints(CMT.votes[:, :2], im_gray)  # blue
		util.draw_keypoints(CMT.outliers[:, :2], im_gray, (222))

		cv2.imshow('main', im_gray)

		# Check key input
		k = cv2.waitKey(1)
		key = chr(k & 255)
		if key == 'q':
			sys.exit()
		if key == 'd':
			import ipdb; ipdb.set_trace()

		# Remember image
		im_prev = im_gray

		# Advance frame number
		frame += 1

		print( '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame))
	return False

################
Gst.init(None)

CLI2="ksvideosrc device-index=0 ! video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 ! appsink name=sink" ## 60 FPS
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