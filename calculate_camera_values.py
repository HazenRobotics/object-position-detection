import cv2 as cv
import numpy as np
import glob

chessboardSize = (9,6)
frameSize = (640,480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('*.jpg')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

image = "WIN_20220111_21_37_44_Pro.jpg"
img = cv.imread(image)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (9,6),None)
corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
ret, rotation_vector, translation_vector = cv.solvePnP(objp, corners2, cameraMatrix, dist)

rotation_matrix = cv.Rodrigues(rotation_vector)
rotation_matrix = rotation_matrix[0]

cv.drawFrameAxes(img, cameraMatrix, dist, rotation_vector, translation_vector, 150)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

print("TRANSLATION_VECTOR: ", translation_vector)
print("ROTATION_MATRIX: ", rotation_matrix)
print("CAMERA_MATRIX: ", cameraMatrix)
