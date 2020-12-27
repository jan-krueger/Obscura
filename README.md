# Obscura

This simply python script allows you to calibrate a camera using the checkerboard calibration pattern. It generates the
a file that contains the camera matrix, and the distortion coefficients. I made this tool because most others that I found
were either not working at all, or generated weird errors, to outright just causing segfaults.

This supports the pinhole and fisheye calibration methods provided by OpenCV. It was developed to be used on Linux, so 
compatibility with any other OS is just coincidental.
## Example

### Live Camera Feed

Suppose that you have connected your camera to your computer. It has `3` *inner* corners length, and height wise. You 
are using a `pinhole` camera, and the side length of each square is `50 [mm]`, and the device is available at `/dev/video0`
then you need to run the following command to start calibration:

`python obscura.py  --rows 3 --columns 3 --model pinhole --squareLength 50 --device 0 `

### Image Frames

Suppose that you have already taken a bunch of pictures with the calibration pattern in it. In that case, you can supply
the `--images` and `--ext` arguments to tell the software where to find the images, and what the file extension is.

`python obscura.py  --rows 3 --columns 3 --model pinhole --squareLength 50 --images ./cal_images --ext png`

This will take all images in the `./cal_images` directory that have a `png` extension.

## Usage
```
usage: obscura.py [-h] --rows ROWS --columns COLUMNS --model {pinhole,fisheye} --squareLength SQUARELENGTH [--frameTime FRAMETIME] [--device DEVICE] [--images IMAGES] [--ext EXT] [--output OUTPUT]

Calibrate a camera using OpenCV.

optional arguments:
  -h, --help            show this help message and exit
  --rows ROWS, -r ROWS  Rows on the chessboard
  --columns COLUMNS, -c COLUMNS
                        Columns on the chessboard
  --model {pinhole,fisheye}, -m {pinhole,fisheye}
  --squareLength SQUARELENGTH, -sl SQUARELENGTH
                        Side length of one square [mm]
  --frameTime FRAMETIME, -ft FRAMETIME
                        Time between capturing images [s]. If set to '-1', you can use the 's' key to take pictures manually.
  --device DEVICE, -d DEVICE
                        Selects the id of the device that should be used to capture the images.
  --images IMAGES, -i IMAGES
                        Directory with images.
  --ext EXT, -e EXT     File extension of images.
  --output OUTPUT, -o OUTPUT
                        Output file name.
```

# Installation

1. Clone this repo `git clone https://github.com/waterfl0w/Obscura`
2. Install numpy and opencv2 
3. Enjoy :)