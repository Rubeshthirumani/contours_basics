import cv2
import imutils


i = cv2.imread('coins.jpg')

img=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)


gaussian_blur=cv2.GaussianBlur(img,(11,11),20)


ret, thresh1 = cv2.threshold(gaussian_blur,245,255,cv2.THRESH_BINARY)



cv2.imshow("Thresholded image",thresh1)

cv2.waitKey()


cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)[1:]




output = i.copy()

cnts_draw = cv2.drawContours(output, cnts, -1, (255, 0, 0), 5)

cv2.imshow("Contours", cnts_draw)
cv2.waitKey(0)
cv2.imwrite("coins_cnts.jpg", cnts_draw)