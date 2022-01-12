import cv2 as cv
import numpy as np
from numpy.linalg import inv
import glob

cam_top_left = (0, 0)
cam_top_right = (1280, 0)
cam_bottom_left = (0, 720)
cam_bottom_right = (1280, 720)

world_top_left = (-63.25, -30, 1)
world_top_right = (69, -28, 1)
world_bottom_left = (-13, -7, 1)
world_bottom_right = (13.5, -7, 1)


#src_pts = np.float32([cam_top_left, cam_top_right, cam_bottom_left, cam_bottom_right])

#dst_pts = np.float32([world_top_left, world_top_right, world_bottom_left, world_bottom_right])

#homography_matrix = cv.findHomography(src_pts, dst_pts)

duck_image_pos = np.array([[330], [85], [1]])

#calculated_duck_pos = np.matmul(homography_matrix[0], duck_image_pos)

#calculated_duck_pos *= (1 / calculated_duck_pos[2][0])

#print("calculated duck pos: ", calculated_duck_pos)

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

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
        # cv.imshow('img', img)
        #cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# img = cv.imread('WIN_20220106_15_19_41_Pro.jpg')
# h,  w = img.shape[:2]
# cameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp * size_of_chessboard_squares_mm

image = "WIN_20220111_21_37_44_Pro.jpg"
img = cv.imread(image)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (9,6),None)

corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv.solvePnP(objp, corners2, cameraMatrix, dist)
print("Box rvec: ", rvecs)
print("Box tvec: ", tvecs)
# cv.drawFrameAxes(img, cameraMatrix, dist, rvecs, tvecs, 100)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

rotation_matrix = cv.Rodrigues(rvecs)
rotation_matrix = rotation_matrix[0]
print("Rotation matrix: ", rotation_matrix)
print("Camera matrix: ", cameraMatrix)

Z_CONST = 1

left_side_mat = inv(rotation_matrix) @ inv(cameraMatrix) @ duck_image_pos
right_side_mat = inv(rotation_matrix) @ tvecs

s = (Z_CONST + right_side_mat[2][0])/left_side_mat[2][0]

calculated_duck_pos = inv(rotation_matrix) @ ( np.subtract(s * inv(cameraMatrix) @ duck_image_pos, tvecs) )
#calculated_duck_pos = homography_matrix[0] @ duck_image_pos
#calculated_duck_pos *= (1 / calculated_duck_pos[2][0])
calculated_duck_pos /= 25.4

print("calculated duck pos: ", calculated_duck_pos)