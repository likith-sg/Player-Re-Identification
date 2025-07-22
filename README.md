
# Player Re-Identification

## Objective

This notebook performs player re-identification on a 15-second sports video clip. The goal is to:
- Detect players using a YOLOv11 object detection model.
- Assign unique IDs to each player.
- Re-identify players who leave the frame and return later, maintaining consistent IDs.
- Simulate real-time tracking and re-identification.

## Demo

### Input (15-second Clip Preview)
![Input](assets/15sec_input_720p.gif)

### Output (Tracked Players)
![Output](assets/output_tracked.gif)

## Environment Setup

This notebook is designed to run on Google Colab or any Python 3.9+ environment. GPU acceleration (CUDA) is recommended for faster execution.
The notebook is structured with clearly labeled markdown cells to enhance readability and guide users through each step. If you prefer to execute the notebook in your local environment, please follow the setup instructions outlined below to configure dependencies and paths accordingly.

### Step 1: Mount Google Drive (Colab only)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Set Working Directory and Paths

```python
WORKDIR = '/content/drive/MyDrive/Player_Re_Identification_Assignment'

MODEL_PATH = f"{WORKDIR}/best.pt"
VIDEO_PATH = f"{WORKDIR}/15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = f"{WORKDIR}/output_tracked.mp4"
```

## Dependencies

Install the required packages using pip:

```bash
pip install ultralytics opencv-python-headless scipy
```

### Python Packages Used

- `ultralytics`: YOLOv11 object detection
- `opencv-python-headless`: Video frame processing
- `scipy`: Cosine similarity for re-identification
- `torch`: GPU acceleration and model inference
- `numpy`, `collections`, `os`: Standard utilities

## Running the Notebook

1. Mount Google Drive and set the working directory.
2. Install all dependencies.
3. Load the YOLOv11 model using:

```python
_load_yolo_model()
```

4. Start the player re-identification process with:

```python
output = process_reid_video(VIDEO_PATH)
```

This function:
- Reads video frame-by-frame
- Detects players using YOLOv11
- Assigns and updates player IDs using IoU and feature matching
- Saves the output video with annotated bounding boxes and player IDs

## Output

- The re-identified and annotated video will be saved to the path specified by `OUTPUT_VIDEO_PATH`.
- Console output includes:
  - Frame-wise statistics
  - Number of new and re-identified players
  - Total number of unique IDs
  - Average number of active players per frame
  - Final summary of detections and performance

## Notes

- The script automatically detects GPU support via PyTorch and uses it if available.
- Uses color histogram features for player re-identification.
- Limits re-identification to a maximum of 22 unique player IDs.
