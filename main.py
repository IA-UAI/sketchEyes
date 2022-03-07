import sys
import argparse
import codecs
import time
import numpy as np
import cv2
from tensorflow import keras

from processing import get_sketch, putText

# Predict class based on gray image
def evaluate(model, sample):
    # Preprocess sample
    sample = cv2.resize(sample, (28, 28), interpolation=cv2.INTER_AREA)
    sample = np.expand_dims(
        sample[:, :, None][:, :, [0]], axis=0
    )  # Prepare sample for prediction
    # Get prediction
    pred = model.predict(sample)[0]
    idx = (-pred).argsort()[:5][0]
    return idx, pred[idx]


def model_init(model_path, classes_path):
    model = keras.models.load_model(model_path)
    classes = []
    with codecs.open(classes_path, encoding="utf-8") as f:
        for line in f:
            classes.append(line.rstrip().capitalize())
    return model, classes


def main(args):
    model, classes = model_init(args.model, args.classes)

    # Initialize Webcam
    cap = cv2.VideoCapture(args.camera_id)

    loop(cap, model, classes)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def loop(cap, model, classes):
    # First run, initialization of variables
    _, frame = cap.read()
    height = int(frame.shape[0])
    width = int(frame.shape[1])

    detection_box_size = 150
    preview_size = 150

    sample = frame[
        height // 2 - detection_box_size : height // 2 + detection_box_size,
        width // 2 - detection_box_size : width // 2 + detection_box_size,
    ]
    sample = get_sketch(sample)

    target_idx = 0
    pred_idx, pred_prob = evaluate(model, sample)

    win_scale = 1.5
    help_enabled = True

    last_time = time.time()
    valid = False
    # Program Loop
    while True:
        # stop if the prediction is valid
        if valid:
            time.sleep(10)

        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)  # FLip frame on the y-axis (mirrors capture)

        # Get sample for prediction
        sample = frame[
            height // 2 - detection_box_size : height // 2 + detection_box_size,
            width // 2 - detection_box_size : width // 2 + detection_box_size,
        ]
        sample = get_sketch(sample)

        # Get prediction
        if time.time() - last_time > 0.3:
            last_time = time.time()
            pred_idx, pred_prob = evaluate(model, sample)

        if target_idx == pred_idx:
            color = (0, 255, 0)  # green
            valid = True
        else:
            color = (0, 0, 255)  # red
            valid = False

        ## Edit Frame
        # Add sample preview window in the corner
        frame[0:preview_size, 0:preview_size, :] = (
            cv2.resize(
                np.copy(sample),
                (preview_size, preview_size),
                interpolation=cv2.INTER_AREA,
            )[:, :, None][:, :, [0, 0, 0]]
            * 255
        )
        # Draw detection box
        cv2.rectangle(
            frame,
            (width // 2 - detection_box_size, height // 2 - detection_box_size),
            (width // 2 + detection_box_size, height // 2 + detection_box_size),
            color,
            2,
        )
        # Add Text
        frame = putText(
            frame,
            f"{pred_prob*100.0:.1f}% {classes[pred_idx]}",
            (width // 2 - detection_box_size, height // 2 + detection_box_size),
            color,
            20,
        )
        text = f"Objetive: {classes[target_idx]}\n\n"
        if help_enabled:
            text = (
                text
                + "Controls:\n- C: To close/open controls\n- A/D: Previous/Next class\n- R: Random Class\n- W/S: Detection Box Size\n- +/-: Window Size\n- P: Pause\n- S: Screenshot\n- Q: Quit"
            )
        frame = putText(frame, text, (10, preview_size + 10), color, 18)
        # Resize frame
        frame = cv2.resize(
            frame, None, fx=win_scale, fy=win_scale, interpolation=cv2.INTER_CUBIC
        )
        # Display the resulting frame
        cv2.imshow("frame", frame)

        # Controls
        opt = cv2.waitKey(2) & 0xFF

        if opt == ord("c"):  # next class
            help_enabled = not help_enabled
        if opt == ord("q"):  # Quit - breaks main loop
            break
        if opt == ord("p"):  # Pause
            while True:
                if cv2.waitKey(2) & 0xFF == ord("s"):
                    cv2.imwrite("frame.png", frame)  # Take screenshot
                else:
                    break

        if (
            opt == ord("w") and detection_box_size < height
        ):  # Increase detection box size
            detection_box_size += 10
        if opt == ord("s") and detection_box_size > 10:  # Decrease detection box size
            detection_box_size -= 10

        if opt == ord("r"):  # rand class
            target_idx = np.random.randint(len(classes))
        if opt == ord("a") and target_idx > 0:  # previous class
            target_idx -= 1
        if opt == ord("d") and target_idx < len(classes) - 1:  # next class
            target_idx += 1

        if opt == ord("+"):  # Increase detection box size
            win_scale += 0.1
        if opt == ord("-") and win_scale > 0.2:  # Decrease detection box size
            win_scale -= 0.1


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("-m", "--model", required=True, help="Keras model file path")
    p.add_argument(
        "-c", "--classes", required=True, help="Classes names file path (.txt)"
    )
    p.add_argument(
        "-cap",
        "--camera-id",
        type=int,
        default=0,
        help="Camera id to be used (default: %(default)s)",
    )
    args = p.parse_args()

    main(args)
