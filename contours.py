import cv2
import imutils
import numpy

img=cv2.imread("shapes.png")

imgshow=cv2.imshow("Image", img)
cv2.waitKey(0) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayshow=cv2.imshow("GrayImage", gray)
cv2.waitKey(0) 
thresh_inv = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]	
cv2.imshow("Thresh", thresh_inv)
cv2.waitKey(0)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]	
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
adaptiveThreshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
	cv2.THRESH_BINARY,231,1)
cv2.imshow("Thresh", adaptiveThreshold)
cv2.waitKey(0)
cnts = cv2.findContours(adaptiveThreshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (0, 0, 255), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)

	#text = "{} objects!".format(len(cnts))
	#cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7,
	#(0, 20, 200), 2)
	#cv2.imshow("Contours", output)
	#cv2.waitKey(0)