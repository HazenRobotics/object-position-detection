import cv2 as cv
import numpy as np

cam_top_left = (0, 0)
cam_top_right = (1280, 0)
cam_bottom_left = (0, 720)
cam_bottom_right = (1280, 720)

world_top_left = (-63.25, -30)
world_top_right = (69, -28)
world_bottom_left = (-13, -7)
world_bottom_right = (13.5, -7)


src_pts = np.float32([cam_top_left, cam_top_right, cam_bottom_left, cam_bottom_right])

dst_pts = np.float32([world_top_left, world_top_right, world_bottom_left, world_bottom_right])

homography_matrix = cv.findHomography(src_pts, dst_pts)

duck_image_pos = np.array([[1280/2], [190], [1]])

calculated_duck_pos = np.matmul(homography_matrix[0], duck_image_pos)

calculated_duck_pos *= (1 / calculated_duck_pos[2][0])

print("calculated duck pos: ", calculated_duck_pos)