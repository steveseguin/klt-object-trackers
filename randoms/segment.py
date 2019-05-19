import numpy as np
import cv2
import sys, getopt

## args
gridx=1140
gridy=912
gridx=440
gridy=212
img = cv2.imread("grid.png")
###
y0,x0,z0 = np.shape(img)
print "input image has dimensions of",x0,y0,z0
ret,thresh = cv2.threshold(img[:,:,0],1,255,0)
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]
#frames = np.array([])
#while len(contours)>0:
#frame = np.zeros((gridx,gridy))
used = np.zeros((len(contours),2), dtype=bool)
print len(contours), "blobs were detected. Cycling thru them now...."
for pos1, cnt1 in enumerate(contours):
	if used[pos1,0]==True: ## Object already added to a frame (as a sub-frame previously I would suspect
			continue ## SKIP
	used[pos1,0]=True
	x1,y1,w1,h1 = cv2.boundingRect(cnt1)
	xmax=x1+w1
	ymax=y1+h1
	cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,255,0),3) ## main object before being shifted around
	
	
	if (x1+gridx>x0): ## if too close to edge of screen, readjust x,y to not go out of bounds
		x1=x0-gridx
	if (y1+gridy>y0): ##
		y1=y0-gridy
	
	## x1, y1 is now potentially slightly different. w1, h1 is not obsolete for some aspects of the code. just keep this in mind
	
	if ((w1>gridx) | (h1>gridy)): ## primary item too big
		cv2.rectangle(img,(x1,y1),(x1+gridx,y1+gridy),(0,0,255),2) ## printing frame  ## This needs to be fixed; what if too large? It should break it up. ah. see next step.
		###
		imgbest = img.copy()
		minoverlap = 999999999
		minwindows = 999999999
		bestxo = 0
		bestyo = 0
		for xo in range(0,gridx):
			print "xo",xo
			for yo in range(0,gridy):
				###  how about we try to break the image up into the least number of cuts; segment them, and then do this. perhaps, even before starting the main countour loop
				totaloverlap=0
				totalwindows=0
				imgnow = img.copy()
				
				for a in range(x1-xo,x1+w1+1,gridx):
					aa=a
					if a+gridx>=x0:
						#print "out of bounds"
						continue
					if a<0:
						aa=0
						#print "out of bounds"
						#continue
					for b in range(y1-yo,y1+h1+1,gridy):
						bb=b
						if b+gridy>=y0:
							#print "out of bounds"
							continue
						if b<0:
							bb=0
							#print "out of bounds"
							#continue
						#############
						
						if 255 in img[bb:b+gridy,aa:a+gridx]:  ## This is prone to cause double overlap. will need to rethink
							totalwindows+=1
							cv2.rectangle(imgnow,(aa,bb),(a+gridx,b+gridy),(0,255,0),2)  ## I need to break it up into chunks; not this bull shit. imgcopy also takes too long.
						#############
				if totalwindows < minwindows:
					minwindows = totalwindows
					bestxo=xo
					bestyo=yo
					imgbest = imgnow.copy()
					print minwindows, bestxo, bestyo
		img = imgbest.copy()
		continue
		print "too large"
		## Break up large contour into pieces?
		# run below contours2  as a function of this peice broken up. if doesn't need to be broken up, then all good.
		## Run thru a loop to find where to cut the object to create the fewest number of bumped pixels that can fit into the grid pattern
			## cutting at the weakest point may make the structure weaker actually, but at least it will be prettier
	else:
		cv2.rectangle(img,(x1,y1),(x1+gridx,y1+gridy),(0,255,0),2) ## printing frame  ## This needs to be fixed; what if too large? It should break it up. ah. see next step.
	for pos2, cnt2 in enumerate(contours):
		if used[pos2,0]==True: ## Object already added to a frame or is current frame. SKIP.
			continue
		x2,y2,w2,h2 = cv2.boundingRect(cnt2)
		if (((x2+w2)-x1>gridx) | ((y2+h2)-y1>gridy)): ## secondary item doesn't fit; too long or too tall; see if it can fit on its own later. Delay this blob		
			print pos1,pos2,"too big",x1,y1
			continue
		
		elif ((x1>x2) | (y1>y2) | (x2>x1+gridx) | (y2>y1+gridy)): ## starting point outside scope of grid; should be delayed. 
		####### I NEED TO SEE IF I CAN SHIFT THINGS AROUND; PERHAPS TRY THINGS BOTH WAYS BEFORE GIVING UP
			print pos1,pos2,"can we shift it? ",x1,y1
			if ((xmax-gridx<=x2) & (ymax-gridy<=y2)):
				print pos1,pos2,"says we can? ",x2,y2
				cv2.rectangle(img,(x1,y1),(x1+gridx,y1+gridy),(20,20,20),2)
				if x1>x2:
					x1=x2
				if y1>y2:
					y1=y2
				if x2+w2>xmax:
					xmax = x2+w2
				if y2+h2>ymax:
					ymax = y2+h2
				if (x1+gridx>x0): ## if too close to edge of screen, readjust x,y to not go out of bounds
					x1=x0-gridx
				if (y1+gridy>y0): ##
					y1=y0-gridy
				cv2.rectangle(img,(x1,y1),(x1+gridx,y1+gridy),(255,255,255),4) ## printing frame  ## This needs to be fixed; what if too large? It should break it up. ah. see next step.
				used[pos2,0]=True
			else:
				print pos1,pos2,"outside scope",x1,y1
				continue

		else:  # ONLY GOOD OUTCOME
			print "GOOD"
			cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(0,255,255),1) ## sub-object added to frame.  Yellow?
			used[pos2,0]=True
		cv2.imshow("img",img)
		cv2.waitKey(1)		
	# if w or h > gridx or gridy, ... 
cv2.imshow("img",img)
cv2.waitKey(0)