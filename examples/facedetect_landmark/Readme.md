# ULFFD with Landmark
## Usage
1. Download `nncase`
```bash
git clone https://github.com/kendryte/nncase.git
```
2. Compile your program and run.
Link to your KD233 development board.
**NOTE** SDK version needs to be greater than [kendryte-standalone-sdk-develop](https://github.com/kendryte/kendryte-standalone-sdk/tree/develop) `2a145b0cacd7123616232ae15ead826d8a83771b`
```bash
cmake .. -DPROJ=facedetect_landmark_example
make
kflash facedetect_landmark_example.bin -B kd233 -p /dev/ttyUSB0 -b 2000000 -t
```
## Result
![demo](demo.gif)

## Credits
[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
[RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)