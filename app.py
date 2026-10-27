"""Interactive Gradio app for handwritten digit prediction."""
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.data import MEAN, STD
from src.model import CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_cnn.pth"
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def prepare_digit(img_array):
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    arr = np.asarray(img_array).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3:4] / 255.0
        arr = arr[..., :3] * alpha + 255.0 * (1 - alpha)
    ink = arr[..., :3].mean(axis=2) if arr.ndim == 3 else arr
    if ink.mean() > 127:
        ink = 255 - ink
    ys, xs = np.where(ink > 40)
    if len(ys) == 0:
        return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = Image.fromarray(ink.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(resized)
    tensor = torch.from_numpy(canvas / 255.0)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0).unsqueeze(0)


def predict_drawing(canvas):
    if canvas is None:
        return None
    if isinstance(canvas, dict):
        canvas = canvas.get("composite")
        if canvas is None:
            return None
    tensor = prepare_digit(canvas)
    if tensor is None:
        return None
    with torch.no_grad():
        probs = torch.softmax(model(tensor.to(DEVICE)), dim=1)[0].cpu().numpy()
    return {str(i): float(probs[i]) for i in range(10)}


demo = gr.Interface(
    fn=predict_drawing,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)", type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Handwritten Digit Recognition",
    description="Draw a digit and the CNN will predict it.",
    live=True,
)

if __name__ == "__main__":
    demo.launch()
