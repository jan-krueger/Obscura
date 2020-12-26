# Import required modules 
import cv2
import numpy as np
import glob
import yaml
import time
import argparse
import tempfile

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def capture_images(device_id, _checkerboard, delta_frame: float, _criteria, _temp_directory):
    """
    :param _temp_directory:
    :param _criteria:
    :param _checkerboard:
    :param delta_frame:
    :type device_id: int Device id of the camera.
    """
    current = time.time()
    video_capture = cv2.VideoCapture(device_id)
    if not video_capture.isOpened():
        print("Failed to open the device.")
        return
    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, _checkerboard,
                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret is True and time.time() - current >= delta_frame:
            current = time.time()
            corners2 = cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1), _criteria)
            frame = cv2.drawChessboardCorners(frame,
                                              _checkerboard,
                                              corners2, ret)
            cv2.imwrite(_temp_directory.name + '/' + str(int(round(time.time() * 1000))) + '.png', original_frame)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(50)

        if key == ord('n'):
            break


def calibrate_camera(images: [], _checkerboard, _square_size, _criteria, width: int = 640, height: int = 480,
                     model: str = 'fisheye') -> object:
    """

    :rtype: object
    """
    if len(images) == 0:
        return False, "No images were found", None

    # 3D points real world coordinates
    objectp3d = np.zeros((1, _checkerboard[0]
                          * _checkerboard[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:_checkerboard[0],
                          0:_checkerboard[1]].T.reshape(-1, 2)
    objectp3d *= _square_size
    threedpoints = []
    twodpoints = []

    counter = 0
    for i in range(len(images)):
        image = cv2.imread(images[i])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the corners in the new image, again
        ret, corners = cv2.findChessboardCorners(image_gray, _checkerboard)

        if ret:
            counter += 1
            threedpoints.append(objectp3d)

            corners2 = cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), _criteria)
            twodpoints.append(corners2)
            image = cv2.drawChessboardCorners(image,
                                              _checkerboard,
                                              corners2, ret)

            cv2.imshow("cropped", image)
            cv2.waitKey(20)
        printProgressBar(i+1, len(images))

    if counter == 0:
        return False, "Not a single image contained a chessboard pattern that could be detected.", None

    matrix = np.zeros((3, 3))
    distortion = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(twodpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(twodpoints))]

    if model == 'fisheye':
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            threedpoints,
            twodpoints,
            image_gray.shape[::-1],
            matrix,
            distortion,
            rvecs,
            tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    elif model == 'pinhole':
        cv2.calibrateCamera(threedpoints, twodpoints, (image_gray.shape[1], image_gray.shape[0]), matrix, distortion,
                            rvecs=rvecs, tvecs=tvecs,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    data = {'camera_matrix': np.asarray(matrix).tolist(),
            'dist_coeff': np.asarray(distortion).tolist()}

    total_error = 0
    for i in range(len(threedpoints)):
        imgpoints2, _ = cv2.projectPoints(threedpoints[i], rvecs[i], tvecs[i], matrix, distortion)
        total_error += cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    return (data, total_error / len(threedpoints), counter / len(images))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Calibrate a camera using OpenCV.')
    parser.add_argument('--rows', '-r', type=int, required=True, help='Rows on the chessboard')
    parser.add_argument('--columns', '-c', type=int, required=True, help='Columns on the chessboard')
    parser.add_argument('--model', '-m', type=str, required=True, choices=['pinhole', 'fisheye'])
    parser.add_argument('--squareLength', '-sl', type=float, required=True, help='Side length of one square [mm]')
    parser.add_argument('--frameTime', '-ft', type=float, default=1, help='Time between capturing images [s]')
    parser.add_argument("--device", "-d", type=int,
                        help='Selects the id of the device that should be used to capture the images.')
    parser.add_argument("--images", "-i", type=str, default='.', help='Directory with images.')
    parser.add_argument('--ext', "-e", type=str, help="File extension of images.")
    parser.add_argument('--output', "-o", type=str, help="Output file name.")

    arguments = parser.parse_args()

    # Define the dimensions of checkerboard
    checkerboard = (arguments.columns, arguments.rows)
    square_size = arguments.squareLength  # [mm]
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    deltaFrame = arguments.frameTime  # [s]

    image_dir = arguments.images
    images = []
    if arguments.device is not None:
        temp_directory = tempfile.TemporaryDirectory()
        capture_images(arguments.device, checkerboard, deltaFrame, criteria, temp_directory)

        if image_dir is not None:
            print("Obscure >> The 'images' argument was provided but the images have been captured using the camera "
                  "therefore the argument will be ignored, and the captured frames from the camera will be used for"
                  " calibration. ")
            images = glob.glob((temp_directory.name + '/') + '*.png')
            temp_directory.cleanup()

    images = glob.glob('%s*.%s' % (image_dir, arguments.ext or "png"))
    data, error, used_images = calibrate_camera(images, checkerboard, square_size, criteria,
                                                width=640, height=480, model=arguments.model)

    if data is False:
        print("Calibration Failed!")
        print("Description: ", error)
    else:
        filename = "calibration_matrix_" + str(int(round(time.time() * 1000))) + ".yaml"
        if arguments.output is not None:
            filename = arguments.output

        print("Used images: ", used_images * 100, "%")
        print("Error: ", error)
        print("Saving calibration data to ", filename)
        # and save it to a file
        with open(filename, "w+") as file:
            yaml.dump(data, file)


if __name__ == "__main__":
    main()
