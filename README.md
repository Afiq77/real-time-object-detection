# Real-Time Object Detection with TensorFlow & OpenCV

This project demonstrates real-time object detection using a pre-trained Convolutional Neural Network (CNN) model from TensorFlow and OpenCV. It identifies and labels multiple objects in live webcam feed, video input, or static images with bounding boxes.

Built to showcase practical applications of deep learning and computer vision, the project is deployable as a FastAPI service or Streamlit demo app.

---

## Features

- Real-time object detection via webcam or video file
- Uses TensorFlow’s SSD MobileNet or COCO-trained model
- Detects common objects like people, vehicles, animals, etc.
- Draws labeled bounding boxes with confidence scores
- Option to deploy as a FastAPI microservice or desktop app
- Docker-ready for portable deployment

---

## Tech Stack

- Python
- TensorFlow
- OpenCV
- NumPy
- FastAPI or Streamlit
- Docker (optional)

---

## Project Structure

    real-time-object-detection/
    │
    ├── app/ # FastAPI or Streamlit frontend
    ├── model/ # TensorFlow saved model (SSD/MobileNet)
    ├── utils/ # Detection logic, preprocessing
    ├── test_videos/ # Sample video files for testing
    ├── requirements.txt
    ├── README.md
    └── .gitignore


---

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/real-time-object-detection.git
    cd real-time-object-detection
    pip install -r requirements.txt
    # COCO SSD MobileNet v2 model (example)
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
    # Streamlit version
    streamlit run app/main.py

    # OR FastAPI version
    uvicorn app.main:app --reload
    Sample Output
    <img src="https://user-images.githubusercontent.com/placeholder/object-detection-demo.gif" alt="demo" width="600"/>

## Future Improvements

- Add webcam toggle in UI
- Optimize inference using TensorRT
- Export annotated video with detection overlay
- Add YOLOv8 or custom-trained model support
- Improve multi-object tracking accuracy

# License
  This project is licensed under the MIT License.


