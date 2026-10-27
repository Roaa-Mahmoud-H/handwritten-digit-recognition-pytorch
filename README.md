# Handwritten Digit Recognition using Neural Networks

A PyTorch project that recognizes handwritten digits (0–9) from the MNIST dataset. It compares a fully connected Multi-Layer Perceptron (MLP) with a Convolutional Neural Network (CNN), evaluates test performance, examines classification errors, and provides an interactive Gradio drawing demo.

## Results

Results below are from the included notebook's recorded run. Training again can produce slightly different values depending on the runtime and software versions.

| Model | Test accuracy | Parameters |
|---|---:|---:|
| MLP | 98.24% | 235,146 |
| CNN | **99.29%** | 421,834 |

The CNN misclassified 71 of the 10,000 MNIST test images in that run. The notebook also reports precision, recall, F1-score, a confusion matrix, and misclassified examples.

## Features

- MNIST data exploration and class-distribution visualization
- Reproducible 55,000 / 5,000 train-validation split, plus the standard 10,000-image test set
- Training-only random affine augmentation
- MLP baseline and CNN with Batch Normalization and Dropout
- Adam optimizer, Cross-Entropy Loss, and StepLR scheduler
- Best-checkpoint selection using validation accuracy
- Test-set evaluation and error analysis
- Preprocessing for uploaded handwritten digits
- Gradio sketchpad demo

## Repository structure

```text
handwritten-digit-recognition-pytorch/
├── Handwritten_Digit_Recognition_using_NN.ipynb  # Full exploratory notebook
├── src/
│   ├── __init__.py
│   ├── data.py                                    # MNIST transforms and loaders
│   └── model.py                                   # MLP and CNN definitions
├── train.py                                       # Train both models and save weights
├── app.py                                         # Interactive Gradio app
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/YOUR_USERNAME/handwritten-digit-recognition-pytorch.git
cd handwritten-digit-recognition-pytorch
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can differ depending on CPU/GPU and operating system. If the default install does not suit your hardware, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Train the models

```bash
python train.py
```

The script downloads MNIST into `data/`, trains MLP and CNN for 10 epochs, evaluates each best validation checkpoint on the test set, and saves `mlp_best.pth`, `cnn_best.pth`, and `mnist_cnn.pth`. These generated files are excluded from Git by `.gitignore`.

## Run the interactive demo

Train the models first so that `mnist_cnn.pth` exists, then run:

```bash
python app.py
```

Open the local URL printed by Gradio and draw a digit from 0 to 9. The app returns the top predictions and their confidence scores. The local app is not permanently hosted; for a public demo, deploy it separately (for example, using [Hugging Face Spaces](https://huggingface.co/spaces)).

## Run the notebook

Open `Handwritten_Digit_Recognition_using_NN.ipynb` in Jupyter Notebook or Google Colab and execute cells in order. The MNIST dataset is downloaded automatically. The notebook includes the training workflow, learning curves, evaluation, confusion matrix, error analysis, image upload test, and Gradio demo.

## Methods

- **MLP:** flattens each 28×28 grayscale image into 784 features and uses fully connected layers with ReLU and Dropout.
- **CNN:** uses convolutional layers, Batch Normalization, ReLU, max pooling, and Dropout to learn spatial features.
- **Preprocessing:** MNIST normalization (mean `0.1307`, standard deviation `0.3081`). Training images receive light affine augmentation; validation and test images do not.

## Limitations and future work

MNIST test accuracy does not guarantee the same performance on photos or handwriting styles outside MNIST. Uploaded images can differ in contrast, centering, thickness, and background. Future work could improve preprocessing, test on a separate real-world dataset, compare additional CNN architectures, and deploy the demo.

## Author

**Roaa Mahmoud**  
Aspiring AI Engineer | Machine Learning | Deep Learning | Computer Vision
