from cv2 import imread, GaussianBlur, imshow, BORDER_DEFAULT, waitKey, CascadeClassifier, cvtColor, COLOR_BGR2GRAY

# Load the cascade
face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')

img = imread('photo.jpg')

# Convert to grayscale
gray = cvtColor(img, COLOR_BGR2GRAY)

# Detect the faces
face_coors = face_cascade.detectMultiScale(gray, 1.1, 4)

for face in face_coors:

    left = face[0]
    top = face[1]
    right = face[2] + left
    bottom = face[3] + top

    croppedImg = img[top:bottom, left:right]

    img[top:bottom, left:right] = GaussianBlur(GaussianBlur(GaussianBlur(GaussianBlur(GaussianBlur(croppedImg, (15,15), BORDER_DEFAULT),(15,15), BORDER_DEFAULT),(15,15), BORDER_DEFAULT),(15,15), BORDER_DEFAULT), (15,15), BORDER_DEFAULT)

imshow('BLURRED FACE(S)', img)

waitKey()
