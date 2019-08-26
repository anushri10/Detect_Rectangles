import cv2
import numpy as np
import matplotlib as plt 


img_clr = cv2.imread('fill3.png')
img = cv2.imread('fill3.png',cv2.IMREAD_GRAYSCALE)

#resize image
print(img.shape)
scale = 800.0 / img.shape[1]
resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
img_clr = cv2.resize(img, (int(img_clr.shape[1] * scale), int(img_clr.shape[0] * scale)))
#Threshold
kernel = np.ones((1, 3), np.uint8)
im = cv2.morphologyEx(resized, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))
thresh, im = cv2.threshold(resized, 140, 255, cv2.THRESH_BINARY)

#dilation and erosion
kernel = np.ones((1, 3), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2) 
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)  

#Remove elements that are too small to fit/excess noise removal
kernel = np.ones((3, 3), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=1)

#contour detection
# im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
unscale = 1.0 / scale
contours = sorted(contours, key = cv2.contourArea)[:]
c=0
rect_c = 0
if contours != None:
	for contour in contours:
		c = c+1
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        # segment based on area
		print(str(c)+ " Contour area: ", cv2.contourArea(contour))
		if (cv2.contourArea(contour) >= 650 or cv2.contourArea(contour) <=400 ):
			rect_c = rect_c+1
			print("Missed : "+str(rect_c)+" Area of missed: "+ str(cv2.contourArea(contour)))
			continue
		#draw rect of smallest possible size for contour	
		rect = cv2.minAreaRect(contour)
		#print("Detected Contour area: ", cv2.minAreaRect(contour))
		#rect = ((int(rect[0][0] * unscale), int(rect[0][1] * unscale)),(int(rect[1][0] * unscale), int(rect[1][1] * unscale)),rect[2])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(img_clr,[box],0,(0,255,0),thickness=2)
		

cv2.imshow("img",img_clr)
cv2.waitKey(5000)
cv2.destroyAllWindows()