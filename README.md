# SketchEyes

Simple sketch classifier using Tensorflow. Detection works by using a webcam

## Usage

In the terminal:

`$ python main.py -m MODEL -c CLASSES [-cap CAMERA_ID]`

- MODEL: Keras model file path
- CLASSES: Classes names file path (.txt)
- CAMERA_ID: Camera id to be used (default: 0)
### Controls

- C: To close/open controls
- A/D: Previous/Next class\n- R: Random Class
- W/S: Increase/Decrease Detection Box Size
- +/-: Increase/Decrease Window Size
- P: Pause
  - S: Screenshot (only on Pause)
- Q: Quit