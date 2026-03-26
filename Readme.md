```markdown
# Sign Language Detection System

A simple real-time **Sign Language Recognition** system that can detect hand signs using a webcam. 

Built with **MediaPipe** for hand tracking and a custom **Convolutional Neural Network (CNN)** trained using TensorFlow/Keras.

---

## Features

- Real-time hand landmark detection using MediaPipe
- Live sign language prediction with confidence percentage
- Easy dataset collection tool
- Simple and effective CNN model
- Training visualization (accuracy & loss graph)

---

## Supported Signs

Currently the model is trained on **4 signs**. You can see the exact signs in the `my_labels.txt` file.

---

## Project Files

- `dataset_gathering.py` → Collect your own hand sign images
- `train_model.py` → Train the CNN model on your data
- `test_model.py` → Test the model in real-time using webcam
- `my_labels.txt` → List of all sign classes
- `my_model.h5` → Trained model file (optional)
- `training_plot.png` → Training accuracy and loss graph
- `requirements.txt` → All required Python packages
- `data/` → Folder containing your collected images

---

## How to Run the Project

### 1. Install Dependencies

First, create and activate a virtual environment (recommended), then run:

```bash
pip install -r requirements.txt
```

### 2. Collect Dataset (if you want to add more signs)

```bash
python dataset_gathering.py
```

- Press **S** to save the current hand gesture
- Press **Q** to quit

### 3. Train the Model

```bash
python train_model.py
```

This script will train the model and save:
- `my_model.h5` (trained model)
- `my_labels.txt` (class names)
- `training_plot.png` (training graph)

### 4. Test in Real-time

```bash
python test_model.py
```

Show your hand in front of the camera and see the prediction!

---

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- TensorFlow
- scikit-learn
- NumPy
- Matplotlib

All packages are listed in `requirements.txt`.




