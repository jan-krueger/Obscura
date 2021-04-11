import cv2
import numpy as np
import glob
import yaml
import time
import argparse
import logging
import os


def print_progress_bar(iteration, total, prefix='', suffix='', length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
    """
    percent = "{0:.2f}".format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '█' * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def capture_images(device_id, model: str, _checkerboard, delta_frame: float, _square_size: float, _criteria):
    """
    :param model:
    :param _square_size:
    :param _criteria:
    :param _checkerboard:
    :param delta_frame:
    :type device_id: int Device id of the camera.
    """
    current = time.time()
    video_capture = cv2.VideoCapture(device_id)

    # GUI
    coverage_ratio = 0  # just keeps track so that we can draw it on the screen
    coverage_image = None
    display_coverage = False

    # 3D points real world coordinates
    objectp3d = create_object_point(_checkerboard, _square_size)

    threedpoints = []
    twodpoints = []

    if not video_capture.isOpened():
        logging.error("Failed to open the camera.")
        return
    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # create overlay image with correct shape if it does not exist
        if coverage_image is None:
            coverage_image = np.zeros(frame.shape, frame.dtype)

        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, _checkerboard,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret is True and (time.time() - current >= delta_frame or delta_frame == -1):
            current = time.time()

            corners2 = cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), _criteria)
            frame = cv2.drawChessboardCorners(frame, _checkerboard, corners2, ret)

            if delta_frame > 0 or (delta_frame == -1 and cv2.waitKey(50) == ord('s')):
                threedpoints.append(objectp3d)
                twodpoints.append(corners2)

                # draw area where it detected the points
                if corners2 is not None:
                    cols, _ = _checkerboard
                    cv2.fillConvexPoly(coverage_image,
                                       np.array(
                                           [corners2[0], corners2[cols - 1], corners2[-1], corners2[-cols]]).astype(
                                           'int32'),
                                       (255, 0, 0))
                    _, counts = np.unique(coverage_image.reshape(-1, 3), return_counts=True, axis=0)
                    coverage_ratio = (counts[1] / (counts[0] + counts[1]))
            # cv2.imwrite(_temp_directory.name + '/' + str(int(round(time.time()))) + '.png', original_frame)

        # GUI
        if display_coverage:
            frame = cv2.addWeighted(coverage_image, 0.4, frame, 0.6, 0)
            cv2.putText(frame, text="Coverage: %.2f%%" % (coverage_ratio * 100), org=(0, 40),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255))

        button_text = "%s%s%s" % (
            'Calibrate [n] | ' if len(twodpoints) > 0 else '',
            'Capture [s] | ' if delta_frame == -1 else '',
            'Coverage [c]: %s' % ('On' if display_coverage else 'Off')
        )
        cv2.putText(frame, text=button_text, org=(0, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(0, 0, 255), thickness=2)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(50)
        if key == ord('n') and len(twodpoints) > 0:
            break
        elif key == ord('c'):
            display_coverage = not display_coverage

    if len(twodpoints) == 0:
        return False, "No images were captured."

    return calibrate(model, threedpoints, twodpoints, _criteria, coverage_image.shape[0:2])


def extract_images(images: [], _checkerboard, _square_size, _criteria, model: str = 'fisheye') -> object:
    """

    :rtype: object
    """
    if len(images) == 0:
        return False, "No images were found"

    # 3D points real world coordinates
    objectp3d = create_object_point(_checkerboard, _square_size)

    threedpoints = []
    twodpoints = []

    logging.info("Extracting calibration patterns from images...")
    _detected_at_least_one = False
    for i in range(len(images)):
        image = cv2.imread(images[i])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the corners in the new image, again
        ret, corners = cv2.findChessboardCorners(image_gray, _checkerboard)

        if ret:
            _detected_at_least_one = True
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), _criteria)
            twodpoints.append(corners2)

        print_progress_bar(i + 1, len(images))

    if len(twodpoints) == 0:
        return False, "Not a single image contained a chessboard pattern that could be detected."

    return calibrate(model, threedpoints, twodpoints, _criteria, image_gray.shape[::-1])


def calibrate(model, threedpoints, twodpoints, _criteria, image_shape):
    camera_matrix = np.zeros((3, 3))
    distortion = None
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(twodpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(twodpoints))]

    if model == 'fisheye':
        distortion = np.zeros((4, 1))
        ret, _, _, _, _ = cv2.fisheye.calibrate(threedpoints, twodpoints, image_shape, camera_matrix, distortion,
                              rvecs=rvecs, tvecs=tvecs,
                              criteria=_criteria)
    elif model == 'pinhole':
        distortion = np.zeros((5, 1))
        ret, _, _, _, _ = cv2.calibrateCamera(threedpoints, twodpoints, image_shape, camera_matrix, distortion,
                            rvecs=rvecs, tvecs=tvecs)

    data = {'camera_matrix': np.asarray(camera_matrix).tolist(),
            'dist_coeff': np.asarray(distortion).tolist()}

    return data, ret

def init_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    formatter = logging.Formatter("Obscura >> %(message)s")
    sh.setFormatter(formatter)

    def decorate_emit(fn):
        def new(*args):
            levelno = args[0].levelno
            color = '\x1b[0m'
            if levelno == logging.ERROR or levelno == logging.CRITICAL:
                color = '\x1b[31;1m'
            elif levelno >= logging.WARNING:
                color = '\x1b[33;1m'
            elif levelno >= logging.INFO:
                color = '\x1b[32;1m'
            elif levelno >= logging.DEBUG:
                color = '\x1b[35;1m'

            args[0].msg = "{0}{1}\x1b[0m".format(color, args[0].msg)
            return fn(*args)

        return new

    sh.emit = decorate_emit(sh.emit)
    logger.addHandler(sh)


def init_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Calibrate a camera using OpenCV.')
    parser.add_argument('--rows', '-r', type=int, required=True, help='Rows on the chessboard')
    parser.add_argument('--columns', '-c', type=int, required=True, help='Columns on the chessboard')
    parser.add_argument('--model', '-m', type=str, required=True, choices=['pinhole', 'fisheye'])
    parser.add_argument('--squareLength', '-sl', type=float, required=True, help='Side length of one square [mm]')
    parser.add_argument('--frameTime', '-ft', type=float, default=1, help='Time between capturing images [s]. If set '
                                                                          'to \'-1\', you can use the \'s\' key to take'
                                                                          ' pictures manually.')
    parser.add_argument("--device", "-d", type=int,
                        help='Selects the id of the device that should be used to capture the images.')
    parser.add_argument("--images", "-i", type=str, help='Directory with images.')
    parser.add_argument('--ext', "-e", type=str, help="File extension of images.")
    parser.add_argument('--output', "-o", type=str, help="Output file name.")

    arguments = parser.parse_args()

    if arguments.device is None and arguments.images is None:
        logging.error('You need to provide either a --device or a directory with --images.')
        return None

    if arguments.device is not None:
        if os.path.exists('/dev/video%d' % arguments.device) is False:
            logging.error('No --device with the id /dev/video%d exists.' % arguments.device)
            return None

    if arguments.images is not None:
        if os.path.exists(arguments.images) is False:
            logging.error("The provided path for the images directory does not exist: '%s'", arguments.images)
            return None
        if arguments.device is not None:
            logging.warning("The 'images' argument was provided but the images will be captured using the camera "
                            "therefore the argument will be ignored, and the captured frames from the camera will be "
                            "used for calibration. ")

    if arguments.frameTime == -1:
        logging.info("--frameTime has been set to '-1'. Use your 's' key to take pictures.")

    return arguments


def create_object_point(checkerboard, square_size):
    objectp3d = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objectp3d *= square_size
    return objectp3d


def main():
    init_logging()
    arguments = init_arguments()
    if arguments is None:
        return

    # Define the dimensions of checkerboard
    checkerboard = (arguments.columns, arguments.rows)
    square_size = arguments.squareLength  # [mm]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    deltaFrame = arguments.frameTime  # [s]

    image_dir = arguments.images
    if arguments.device is not None:
        data, error = capture_images(arguments.device, arguments.model, checkerboard, deltaFrame, square_size, criteria)
    else:
        images = glob.glob('%s/*.%s' % (image_dir, arguments.ext or "png"))
        data, error = extract_images(images, checkerboard, square_size, criteria, model=arguments.model)

    # calculate coverage based on how many values are not zero
    # display 'n' to continue processing
    # show how many images have been taken

    if data is False:
        logging.error("Calibration Failed! - %s", error)
    else:
        filename = "calibration_matrix_" + str(int(round(time.time() * 1000))) + ".yaml"
        if arguments.output is not None:
            filename = arguments.output

        logging.info("Re-projection error: %.4f", error)
        if error > 10:
            logging.critical("The re-projection error is big - it should be as close to 0 as possible. Try to "
                             "recalibrate with better coverage. A good rule of thumb for calibrating is to translate,"
                             " and rotate the camera on all major axis by making smooth movements, and repeating this"
                             " for each translation and rotation three times.")

        logging.info("Saving calibration data to %s", filename)
        logging.info(data)

        # and save it to a file
        with open(filename, "w+") as file:
            yaml.dump(data, file)


if __name__ == "__main__":
    main()
