import cv2
cascade = cv2.CascadeClassifier("/Users/srx/PycharmProjects/LearningAlgorithms/Face_Recognition/haarcascade_frontalface_default.xml")
img = cv2.imread("/Users/srx/PycharmProjects/LearningAlgorithms/Face_Recognition/Group-Photo.jpg")
resized = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray_img, scaleFactor=1.05,minNeighbors=5)
print(faces)
for x,y,w,h in faces:
    resized = cv2.rectangle(resized, (x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow("Group", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()