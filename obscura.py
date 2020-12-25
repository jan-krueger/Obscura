# Import required modules 
import cv2 
import numpy as np 
import glob
import yaml
import time
import math
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Calibrate a camera using OpenCV.')
parser.add_argument('--rows', '-r', type=int, required=True, help='Rows on the chessboard')
parser.add_argument('--columns', '-c', type=int, required=True, help='Columns on the chessboard')
parser.add_argument('--model', '-m', type=str, required=True, choices=['pinhole',  'fisheye'])
parser.add_argument('--squareLength', '-sl', type=float, required=True, help='Side length of one square [mm]')
parser.add_argument('--frameTime', '-ft', type=float, default=1, help='Time between capturing images [s]')
parser.add_argument("--device", "-d", type=int, help='Selects the id of the device that should be used to capture the images.')
parser.add_argument("--images", "-i", type=str, default='.', help='Directory with images.')
parser.add_argument('--ext', "-e", type=str, help="File extenion of images.")
parser.add_argument('--output', "-o", type=str, help="Output file name.")

arguments = parser.parse_args()

# Define the dimensions of checkerboard 
CHECKERBOARD = (arguments.columns,arguments.rows) 
square_size = arguments.squareLength # [mm]
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
deltaFrame = arguments.frameTime # [s]

def captureImages(deviceId):
	current = time.time()
	video_capture = cv2.VideoCapture(deviceId)
	while True:
		if not video_capture.isOpened():
			print('Unable to load camera.')
			pass

		# Capture frame-by-frame
		ret, frame = video_capture.read()
		original_frame = frame.copy()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE) 

		if ret == True and time.time() - current >= deltaFrame: 
			current = time.time()
			corners2 = cv2.cornerSubPix( 
				gray, corners, (3, 3), (-1, -1), criteria) 
			frame = cv2.drawChessboardCorners(frame, 
											CHECKERBOARD, 
											corners2, ret) 
			cv2.imwrite('images/' + str(int(round(time.time() * 1000))) + '.png',original_frame)

		cv2.imshow('Video', frame)
		key = cv2.waitKey(50)

		if(key == ord('n')):
			break	

def cutout_chessboards(images, factor = 1):
	# Extracting path of individual image stored 
	# in a given directory. Since no path is 
	# specified, it will take current directory 
	# jpg files alone 
	frame_size = ()

	# find the corner points in all images and determine the max size of frame that we need
	grid_positions = []

	for filename in images: 
		image = cv2.imread(filename) 
		grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
		frame_size = image.shape[0:2]
		width = frame_size[1]
		height = frame_size[0]
		ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE) 
		# assert ret
		if ret:
			min_x = width-1
			max_x = 0
			min_y = height-1
			max_y = 0

			for corner in corners:
				c = corner[0]
				x = c[0] 
				y = c[1]

				min_x = min(min_x,x)
				max_x = max(max_x,x)
				min_y = min(min_y,y)
				max_y = max(max_y,y)
			
			minf = 1 - factor
			maxf = 1 + factor

			grid_positions.append([(math.floor(min_x*minf), math.ceil(max_x*maxf)), (math.floor(min_y*minf), math.ceil(max_y*maxf))])
	return grid_positions

def calibrate_camera(images, grid_positions, width = 640, height = 480, cropped = False, model='fisheye'):
	# 3D points real world coordinates 
	objectp3d = np.zeros((1, CHECKERBOARD[0] 
						* CHECKERBOARD[1], 
						3), np.float32) 
	objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
								0:CHECKERBOARD[1]].T.reshape(-1, 2) 
	objectp3d *= square_size
	threedpoints = [] 
	twodpoints = [] 

	counter = 0
	for i in range(len(images)): 
		image = cv2.imread(images[i]) 

		if(cropped):
			dim = grid_positions[i]
			w = dim[0][1]-dim[0][0]
			h = dim[1][1]-dim[1][0]

			# push the cropped image into a uniform image
			blank_image = np.ones((width,height,3), np.uint8) * 255
			try:
				blank_image[dim[1][0]:dim[1][1],dim[0][0]:dim[0][1]] = image[dim[1][0]:dim[1][1],dim[0][0]:dim[0][1]]
			except:
				continue # TODO why does this fail sometimes?
			image = blank_image

		grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

		# find the corners in the new image, again
		ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD) 

		if ret: 
			counter += 1
			threedpoints.append(objectp3d) 

			corners2 = cv2.cornerSubPix(grayColor, corners, (3, 3), (-1, -1), criteria) 
			twodpoints.append(corners2) 
			image = cv2.drawChessboardCorners(image, 
											CHECKERBOARD, 
											corners2, ret) 

			cv2.imshow("cropped", image)
			cv2.waitKey(20)

	matrix = np.zeros((3, 3))
	distortion = np.zeros((4, 1))
	rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(twodpoints))]
	tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(twodpoints))]

	#ret, matrix, distortion, r_vecs, t_vecs = cv2.fisheye.calibrate(threedpoints, twodpoints, (grayColor.shape[1],grayColor.shape[0]), None, None,criteria= (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) 

	if model == 'fisheye':
		print("Using fisheye for calibration")
		rms, _, _, _, _ = cv2.fisheye.calibrate(
			threedpoints,
			twodpoints,
			grayColor.shape[::-1],
			matrix,
			distortion,
			rvecs,
			tvecs,
			cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW,
			(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
		)
	elif model == 'pinhole':
		print("Using pinhole for calibration")
		cv2.calibrateCamera(threedpoints, twodpoints, (grayColor.shape[1], grayColor.shape[0]), matrix, distortion, rvecs=rvecs, tvecs=tvecs, criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6));
	
	data = {'camera_matrix': np.asarray(matrix).tolist(),
			'dist_coeff': np.asarray(distortion).tolist()}

	total_error = 0
	for i in range(len(threedpoints)):
		imgpoints2, _ = cv2.projectPoints(threedpoints[i], rvecs[i], tvecs[i], matrix, distortion)
		error = cv2.norm(twodpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		total_error += error

	return (data,total_error/len(threedpoints),counter/len(images))


if arguments.device:
	captureImages(arguments.device)
images = glob.glob('%s*.%s' % (arguments.images, arguments.ext)) 
grid_positions = cutout_chessboards(images,factor=1)
data,error,used_images = calibrate_camera(images, grid_positions, width=640,height=480, model=arguments.model, cropped=False)

print("Error: ", error)
print("Used images: ", used_images * 100, "%")

# and save it to a file
with open("calibration_matrix_" + str(int(round(time.time() * 1000))) + ".yaml", "w+") as f:
    yaml.dump(data, f)
