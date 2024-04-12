from flask import Flask, render_template, Response
import cv2
import time
from YOLO_Video import video_detection
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def read_from_webcam():
    # time.sleep(10)
    source = 0
    # source = "http://10.10.10.153:4747/video"
    yolo_output = video_detection(source)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/image_feed")
def image_feed():
    return Response(read_from_webcam(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
