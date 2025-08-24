# 🍃 Leaf Detection Model

This project aims to detect and classify leaves using a deep learning model trained on a custom dataset. The pipeline includes data preprocessing, model training, and real-time inference. Built using PyTorch and Jupyter notebooks.

---

## 🚀 Step 1: Setup the Environment

### 🔁 Clone the Repository

```bash
git clone https://github.com/TNKodi/Leaf-Detection.git
cd Leaf-Detection
```

### 🧪 Create Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### ⚙️ Install PyTorch

#### ✅ With CUDA (GPU support)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### ❎ Without CUDA (CPU only)
```bash
pip install torch torchvision torchaudio
```

### 📦 Install Other Required Libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install matplotlib opencv-python pandas seaborn tqdm scikit-learn jupyter
```

---

## 📂 Step 2: Prepare Dataset

- 📥 Download the Kaggle dataset:  
  [**Kaggle Leaf Detection Dataset**](https://www.kaggle.com/datasets/andrewmvd/leaf-detection)

- 📁 Create a folder named `dataset/` in the root directory and extract the dataset inside it:
```
Leaf-Detection/
├── dataset/
│   ├── train/
│   ├── test/
│   └── ...
```

---

## 🧹 Step 3: Run Preprocessing Notebook

Open the preprocessing notebook and run all cells:

```bash
jupyter notebook
```

Then open:
```
📘 01_Data_Preprocessing.ipynb
```

Make sure the dataset is correctly structured inside the `dataset/` folder before running.

---

## 🏋️ Step 4: Train the Model

Open and run:
```
📘 02_Model_Train.ipynb
```

📌 **Important**: You can change the number of training epochs and other hyperparameters in the cell under:

```python
NUM_EPOCHS = 50  # <-- Change this as needed
```

Training results and checkpoints (e.g. best model) will be saved in the `checkpoints/` folder (or similar as per notebook).

---

## 🧪 Step 5: Test & Real-Time Inference

Open the notebook:
```
📘 03_Test_and_Realtime.ipynb
```

### 🔄 Update Model Path

In this notebook, make sure to set the path to your trained model:

```python
MODEL_PATH = "checkpoints/best_model.pth"  # <-- Set your trained model path here
```

Then run the cells to test on sample images or perform real-time inference using a webcam or uploaded images.

---

## 📊 Results

You can find training metrics, confusion matrix, and test predictions inside the notebook outputs.

---

## 🛠️ Project Structure

```
Leaf-Detection/
├── dataset/                 # Dataset folder (user must add)
├── checkpoints/             # Trained model weights (created after training)
├── 01_Data_Preprocessing.ipynb
├── 02_Model_Train.ipynb
├── 03_Test_and_Realtime.ipynb
├── README.md
└── requirements.txt
```

---

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/leaf-detection)

---

## 📌 Notes

- Ensure your system has Python 3.8 or newer.
- Training may take time depending on hardware.
- You can export your trained model to ONNX or TorchScript if needed.

---
