
import numpy as np
import cv2
import sys, getopt

inputfile = 'EiffelBot0771.png'
outputfile = 'output.png'
try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print 'script.py -i <inputfile> -o <outputfile>'
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
	 print 'script.py -i <inputfile> -o <outputfile>'
	 sys.exit()
  elif opt in ("-i", "--ifile"):
	 inputfile = arg
  elif opt in ("-o", "--ofile"):
	 outputfile = arg
print 'script.py -i <inputfile> -o <outputfile>'
print 'Input file is "', inputfile
print 'Output file is "', outputfile

def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
	
    return img[np.ix_(mask.any(1),mask.any(0))]

img = cv2.imread(inputfile)

img = 255-img[:,:,0]
img[np.nonzero(img)]=255
img=255-img
img_original=img

mask = img==255
x_offset,y_offset, x_end, y_end = np.ix_(mask.any(1))[0][0],np.ix_(mask.any(0))[0][0],np.ix_(mask.any(1))[0][-1],np.ix_(mask.any(0))[0][-1]
#img = img_original[x_offset:x_end,y_offset:y_end] ## crop img
#xx,yy =  np.shape(img) ## new crop img size
print "offset is", x_offset,y_offset, x_end, y_end
gridx=1140
gridy=912 # 912*1140

a = cv2.getGaussianKernel(gridy,250)
b = cv2.getGaussianKernel(gridx,250)
c = b*a.T
d = 1-c/c.max()
#print d.max(),d.min()
#cv2.imshow("guassian filter used for distortion weighting",d)
#cv2.waitKey(0)

best=9999999999
bestbump=9999999999
bestquad=9999999999
bestx=0
besty=0
yo = 0
for xo in range(-gridx,0):
	counter=0
	bump=0
	output = img/2
	quad=np.zeros((gridx,gridy),dtype=np.uint16)
	for x in range(xo+x_offset,x_end+1,gridx):
		xxx=x
		if x<0:
			xxx=0
		for y in range(yo+y_offset,y_end+1,gridy):
			yyy=y
			if y<0:
				yyy=0
			if 255 in img[xxx:x+gridx,yyy:y+gridy]:
				output[xxx:xxx+gridx,yyy:yyy+gridy]+=100
				output[xxx:x+gridx,yyy:y+gridy]-=30
				counter+=1
				xs,ys = np.shape(img[xxx:xxx+gridx,yyy:yyy+gridy])
				#print np.shape(img)
				#print xxx-x,yyy-y,xxx,yyy,xs,ys,x,y,xo,yo
				
				quad[xxx-x:xs+xxx-x,yyy-y:ys+yyy-y] += img[xxx:x+gridx,yyy:y+gridy]
				if (yyy+1<y_end):
					bump+= np.sum(np.where((img[xxx:x+gridx,yyy]==255) & (img[xxx:x+gridx,yyy+1]==255) & (img[xxx:x+gridx,yyy]==img[xxx:x+gridx,yyy+1])))
				if (xxx+1<x_end):
					bump+= np.sum(np.where((img[xxx,yyy:y+gridy]==255) & (img[xxx+1,yyy:y+gridy]==255) & (img[xxx,yyy:y+gridy]==img[xxx+1,yyy:y+gridy])))
			
			output[xxx:xxx+gridx,y+gridy:y+gridy+3]=200
			output[x+gridx:x+gridx+3,yyy:yyy+gridy]=200
			output[xxx:xxx+gridx,yyy:yyy+1]=255
			output[xxx:xxx+1,yyy:yyy+gridy]=255
			
	quadscore=np.sum(quad*d)
	print bump,counter,xo+x_offset,yo+y_offset
	if bump<bestbump:
		bestbump=bump
		bestx = xo+x_offset
		bestimg = output
		best=counter
	elif bump==bestbump:
		if counter<best:
			best=counter
			bestx = xo+x_offset
			bestimg = output
		elif counter==best:
			if quadscore<bestquad:
				bestquad=quadscore
				bestx = xo+x_offset
				bestimg = output
				
		
for yo in range(-gridy,0):
	counter=0
	bump=0
	output = img/2
	quad=np.zeros((gridx,gridy),dtype=np.uint16)
	for x in range(bestx,x_end+1,gridx):
		xxx=x
		if x<0:
			xxx=0
		for y in range(yo+y_offset,y_end+1,gridy):
			yyy=y
			if y<0:
				yyy=0
			if 255 in img[xxx:x+gridx,yyy:y+gridy]:
				output[xxx:xxx+gridx,yyy:yyy+gridy]+=100
				output[xxx:x+gridx,yyy:y+gridy]-=30
				counter+=1
				xs,ys = np.shape(img[xxx:xxx+gridx,yyy:yyy+gridy])
				#print np.shape(img)
				#print xxx-x,yyy-y,xxx,yyy,xs,ys,x,y,xo,yo
				
				quad[xxx-x:xs+xxx-x,yyy-y:ys+yyy-y] += img[xxx:x+gridx,yyy:y+gridy]
				if (yyy+1<y_end):
					bump+= np.sum(np.where((img[xxx:x+gridx,yyy]==255) & (img[xxx:x+gridx,yyy+1]==255) & (img[xxx:x+gridx,yyy]==img[xxx:x+gridx,yyy+1])))
				if (xxx+1<x_end):
					bump+= np.sum(np.where((img[xxx,yyy:y+gridy]==255) & (img[xxx+1,yyy:y+gridy]==255) & (img[xxx,yyy:y+gridy]==img[xxx+1,yyy:y+gridy])))
			
			output[xxx:xxx+gridx,y+gridy:y+gridy+3]=200
			output[x+gridx:x+gridx+3,yyy:yyy+gridy]=200
			output[xxx:xxx+gridx,yyy:yyy+1]=255
			output[xxx:xxx+1,yyy:yyy+gridy]=255
				

	quadscore=np.sum(quad*d)
	print bump,counter,xo+x_offset,yo+y_offset
	if bump<bestbump:
		bestbump=bump
		besty = yo+y_offset
		bestimg = output
		best=counter
	elif bump==bestbump:
		if counter<best:
			best=counter
			besty = yo+y_offset
			bestimg = output
		elif counter==best:
			if quadscore<bestquad:
				bestquad=quadscore
				besty = yo+y_offset
				bestimg = output
	#cv2.imshow("output",output)
	#cv2.waitKey(1)
	
# for xo in range(-gridx,0):
	# counter=0
	# bump=0
	# output = img/2
	# quad=np.zeros((gridx,gridy),dtype=np.uint16)
	# for x in range(besty,x_end+1,gridx):
		# xxx=x
		# if x<0:
			# xxx=0
		# for y in range(yo+y_offset,y_end+1,gridy):
			# yyy=y
			# if y<0:
				# yyy=0
			
			# if 255 in img[xxx:x+gridx,yyy:y+gridy]:
				# output[xxx:x+gridx,yyy:y+gridy]+=100
				# counter+=1
				# xs,ys = np.shape(img[xxx:x+gridx,yyy:y+gridy])
				# #print np.shape(img)
				# #print xxx-x,yyy-y,xxx,yyy,xs,ys,x,y,xo,yo
				
				# quad[xxx-x:xs+xxx-x,yyy-y:ys+yyy-y] += img[xxx:x+gridx,yyy:y+gridy]
				# if (yyy+1<y_end):
					# bump+= np.sum(np.where((img[xxx:x+gridx,yyy]==255) & (img[xxx:x+gridx,yyy+1]==255) & (img[xxx:x+gridx,yyy]==img[xxx:x+gridx,yyy+1])))
				# if (xxx+1<x_end):
					# bump+= np.sum(np.where((img[xxx,yyy:y+gridy]==255) & (img[xxx+1,yyy:y+gridy]==255) & (img[xxx,yyy:y+gridy]==img[xxx+1,yyy:y+gridy])))
					
			# output[xxx:x+gridx,yyy:yyy+1]=255
			# output[xxx:xxx+1,yyy:y+gridy]=255
			
	# quadscore=np.sum(quad*d)
	# print bump,counter,xo+x_offset,yo+y_offset
	# if bump<bestbump:
		# bestbump=bump
		# bestx = xo+x_offset
		# bestimg = output
		# best=counter
	# elif bump==bestbump:
		# if counter<best:
			# best=counter
			# bestx = xo+x_offset
			# bestimg = output
		# elif counter==best:
			# if quadscore<bestquad:
				# bestquad=quadscore
				# bestx = xo+x_offset
				# bestimg = output
				
				
				
bestimg= cv2.resize(bestimg, (700,900), interpolation = cv2.INTER_AREA)
print "Number of moves needed is ",best,", with starting x,y offset of: ",bestx,besty
cv2.imshow("bestimg",bestimg)
cv2.waitKey(0)