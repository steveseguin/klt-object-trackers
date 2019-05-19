'''
MOSSE tracking sample
This sample implements correlation-based tracking approach, described in [1].
Usage:
  Draw rectangles around objects with a mouse to track them.
Keys:
  c        - clear targets
[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
	
	
'''

import numpy as np
import cv2
from common import draw_str, RectSelector
import time
import serial

eps = 1e-5  ## MOSSE SETTING?

ser = serial.Serial(7)
ser.baudrate = 9600
anglex = 90

byteone=int(254*anglex/180)
#move to an absolute position in 8-bit mode (0x04 for the mode, 0 for the servo, 0-255 for the position (spread over two bytes))
bud=chr(0xFF)+chr(0)+chr(byteone)
ser.write(bud)

def camera_shift(img, prev_img):
	img = cv2.GaussianBlur(img,(5,5),0)
	prev_img = cv2.GaussianBlur(prev_img,(5,5),0)
	best_deltasum = 999999999
	best_x = 0
	best_y = 0
	###
	for x in range(-30,31):
		static = prev_img[:,30:640-30]
		dynamic = img[:,30+x:640-30+x]

		delta = cv2.absdiff(static, dynamic)
		deltasum = delta.sum()
		
		if best_deltasum > deltasum:
			best_deltasum = deltasum
			best_delta = delta
			best_x = x
	###	
	x=best_x
	for y in range(-20,21):
		static = prev_img[20:360-20,30:640-30]
		dynamic = img[20+y:360-20+y,30+x:640-30+x]

		delta = cv2.absdiff(static, dynamic)
		deltasum = delta.sum()
		
		if best_deltasum > deltasum:
			best_deltasum = deltasum
			best_delta = delta
			best_y = y
	###
	y = best_y
	for x in range(-30,31):
		static = prev_img[20:360-20,30:640-30]
		dynamic = img[20+y:360-20+y,30+x:640-30+x]

		delta = cv2.absdiff(static, dynamic)
		deltasum = delta.sum()
		
		if best_deltasum > deltasum:
			best_deltasum = deltasum
			best_delta = delta
			best_x = x
	###
	#print best_x,best_y,best_deltasum,deltasum
	return best_x,best_y

class Kalman2D(object):
    '''
    A class for 2D Kalman filtering
    '''

    def __init__(self, processNoiseCovariance=1e-1, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1): #  DEFAULT = processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1
		'''
		Constructs a new Kalman2D object.  
		For explanation of the error covariances see
		http://en.wikipedia.org/wiki/Kalman_filter
		'''

		self.kalman = cv2.cv.CreateKalman(4, 2, 0)
		self.kalman_state = cv2.cv.CreateMat(4, 1, cv2.cv.CV_32FC1)
		self.kalman_process_noise = cv2.cv.CreateMat(4, 1, cv2.cv.CV_32FC1)
		self.kalman_measurement = cv2.cv.CreateMat(2, 1, cv2.cv.CV_32FC1)

		for j in range(4):
			for k in range(4):
				self.kalman.transition_matrix[j,k] = 0
			self.kalman.transition_matrix[j,j] = 1

		cv2.cv.SetIdentity(self.kalman.measurement_matrix)

		cv2.cv.SetIdentity(self.kalman.process_noise_cov, cv2.cv.RealScalar(processNoiseCovariance))
		cv2.cv.SetIdentity(self.kalman.measurement_noise_cov, cv2.cv.RealScalar(measurementNoiseCovariance))
		cv2.cv.SetIdentity(self.kalman.error_cov_post, cv2.cv.RealScalar(errorCovariancePost))

		self.predicted = None
		self.esitmated = None

    def update(self, x, y):
		'''
		Updates the filter with a new X,Y measurement
		'''

		self.kalman_measurement[0, 0] = x
		self.kalman_measurement[1, 0] = y
		#self.kalman_measurement[2, 0] = w
		#self.kalman_measurement[3, 0] = h

		self.predicted = cv2.cv.KalmanPredict(self.kalman)
		self.corrected = cv2.cv.KalmanCorrect(self.kalman, self.kalman_measurement)

    def getEstimate(self):
        '''
        Returns the current X,Y estimate.
        '''

        return self.corrected[0,0], self.corrected[1,0]

    def getPrediction(self):
        '''
        Returns the current X,Y prediction.
        '''

        return self.predicted[0,0], self.predicted[1,0]

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C



class MOSSE:
    def __init__(self, frame, rect):
		x1, y1, x2, y2 = rect
		w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
		x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
		self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
		self.size = w, h
		self.kal = Kalman2D()
		self.new_H = True
		img = cv2.getRectSubPix(frame, (w, h), (x, y))

		self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
		g = np.zeros((h, w), np.float32)
		g[h//2, w//2] = 1
		g = cv2.GaussianBlur(g, (-1, -1), 2.0)
		g /= g.max()

		self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
		self.H1 = np.zeros_like(self.G)
		self.H2 = np.zeros_like(self.G)
		for i in xrange(128):
			a = self.preprocess(rnd_warp(img))
			A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
			self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
			self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
		self.H = divSpec(self.H1, self.H2)
		self.H[...,1] *= -1
		self.Hs = [[self.H,self.H1,self.H2]]
		self.update(frame)

    def update(self, frame, x_shift = 0, y_shift = 0, rate = 0.125):  ## DEFAULT LEARNING RATE is 0.125   
		global anglex, ser
		(x, y), (w, h) = self.pos, self.size
		x += x_shift
		y += y_shift
		self.pos = (x, y)

		self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
		img = self.preprocess(img)
		self.last_resp, (dx, dy), self.psr = self.correlate(img)
		self.good = self.psr > 8.0  # didn't find it where it previously was, even with camera compensation
		if not self.good:
			return

		self.pos = x+dx, y+dy
		self.kal.update(dx, dy)

		self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
		img = self.preprocess(img)		
		A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
		H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
		H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
		
		##########
		

		bestscore = -1
		secondbest = -1
		besti = -1

		for i in range(len(self.Hs)):
			C = cv2.mulSpectrums(A, self.Hs[i][0], 0, conjB=True)
			resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
			h, w = resp.shape
			_, mval, _, (mx, my) = cv2.minMaxLoc(resp)
			side_resp = resp.copy()
			cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
			smean, sstd = side_resp.mean(), side_resp.std()
			score = (mval-smean) / (sstd+eps)
			
			if score > bestscore :
				secondbest = bestscore
				bestscore = score
				besti = i


		##############
		
		##############
		
		
		self.H1 = self.H1 * (1.0-rate) + H1 * rate
		self.H2 = self.H2 * (1.0-rate) + H2 * rate
		self.H = divSpec(self.H1, self.H2)
		self.H[...,1] *= -1
		
		if bestscore > 64:
			print "match", besti
			self.H,self.H1,self.H2 = self.Hs[besti][0],self.Hs[besti][1],self.Hs[besti][2]
		elif ((i<20) & (bestscore > 8) & ((bestscore - secondbest) > 32)):
			print "new",i
			self.Hs = np.vstack((self.Hs,[[self.H,self.H1,self.H2]]))
		else:
			print "update",i
			self.Hs[besti][1] = self.Hs[besti][1] * (1.0-rate) + H1 * rate
			self.Hs[besti][2] = self.Hs[besti][2] * (1.0-rate) + H2 * rate
			self.H = divSpec(self.Hs[besti][1], self.Hs[besti][2])
			self.H[...,1] *= -1
			self.Hs[besti][0],self.Hs[besti][1],self.Hs[besti][2] = self.H,self.H1,self.H2
			

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
		(x, y), (w, h) = self.pos, self.size
		x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
		cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
		dxp,dyp = self.kal.getEstimate()
		cv2.circle(vis, (int(dxp+x), int(dyp+y)), 2, (255, 255, 255), -1)
		if self.good:
			cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
		else:
			cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
			cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
		draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)


    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
				
		C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
		resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
		h, w = resp.shape
		_, mval, _, (mx, my) = cv2.minMaxLoc(resp)
		side_resp = resp.copy()
		cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
		smean, sstd = side_resp.mean(), side_resp.std()
		psr = (mval-smean) / (sstd+eps)
		return resp, (mx-w//2, my-h//2), psr
	
    def correlate_all(self, img, HS):
		C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), HS, 0, conjB=True)
		resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
		h, w = resp.shape
		_, mval, _, (mx, my) = cv2.minMaxLoc(resp)
		side_resp = resp.copy()
		cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
		smean, sstd = side_resp.mean(), side_resp.std()
		psr = (mval-smean) / (sstd+eps)
		return resp, (mx-w//2, my-h//2), psr
		

class App:
	def __init__(self):
		#self.cap = cv2.VideoCapture('sailing.mp4')
		#self.cap = cv2.VideoCapture('biker.mp4')
		#self.cap = cv2.VideoCapture('chase.mp4')
		#self.cap = cv2.VideoCapture('park.mp4')
		self.cap = cv2.VideoCapture(0)
		self.cap.set(3,640)
		self.cap.set(4,360)
		self.cap.set(5,30)
		#for fff in range(0,300):
		#		self.cap.grab()
		#self.cap.set(6,cv2.cv.CV_FOURCC('M','J','P','G')) ## self.cap.set(6,cv2.cv.CV_FOURCC('H','2','6','4'))   ## 	
		#self.counter=549		
		#self.cap = cv2.VideoCapture("./runner/img_0000"+str(self.counter).zfill(6)+".png")

		_, self.frame = self.cap.read()
		
		self.prev_gray_blur = cv2.GaussianBlur(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),(5,5),0)
		cv2.imshow('frame', self.frame)
		self.rect_sel = RectSelector('frame', self.onrect)
		self.trackers = []
		self.shift_x, self.shift_y = 0,0
		cv2.waitKey(0)

	def onrect(self, rect):
		frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		tracker = MOSSE(frame_gray, rect)
		self.trackers.append(tracker)

	def run(self):


		while True:
			#start = time.time()
			#self.counter+=1		
			#self.cap = cv2.VideoCapture("./runner/img_0000"+str(self.counter).zfill(6)+".png")
			#self.cap = cv2.VideoCapture(0)
			ret, self.frame = self.cap.read()
			if not ret:
				break
			frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			##
			frame_gray_blur = cv2.GaussianBlur(frame_gray,(5,5),0)
			self.shift_x, self.shift_y = camera_shift(frame_gray_blur, self.prev_gray_blur)
			#print self.shift_x, self.shift_y
			self.prev_gray_blur = frame_gray_blur.copy()
			##
			for tracker in self.trackers:
				tracker.update(frame_gray, self.shift_x, self.shift_y)
			vis = self.frame.copy()
			for tracker in self.trackers:
				tracker.draw_state(vis)
			if len(self.trackers) > 0:
				cv2.imshow('tracker state', self.trackers[-1].state_vis)
			self.rect_sel.draw(vis)
			cv2.imshow('frame', vis)
			ch = cv2.waitKey(10) & 0xFF
			if ch == 27:
				break
			if ch == ord('c'):
				self.trackers = []
			end = time.time()
			#print int(1/(end - start))


			
if __name__ == '__main__':

	App( ).run()
	
	
	
	
	
	
	
	
	
	
	
	