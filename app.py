import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("autism_detector_model_final.h5")

# Define image size (the size you used during training)
IMG_SIZE = (128, 128)

def preprocess_image(inp):
    """
    Accepts PIL.Image or numpy array from Gradio.
    Returns a float32 numpy array shaped (1, H, W, C).
    """
    if isinstance(inp, np.ndarray):
        img = Image.fromarray(inp.astype('uint8'))
    elif isinstance(inp, Image.Image):
        img = inp
    else:
        raise ValueError(f"Unsupported input type: {type(inp)}")

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(IMG_SIZE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img, source):
    """
    Predict autism status, with different responses for webcam or upload.
    """
    if img is None:
        return "No image provided."

    processed = preprocess_image(img)
    pred = model.predict(processed)[0][0]

    # Decide the class
    if pred > 0.5:
        result = "Autistic"
        confidence = pred * 100
    else:
        result = "Non-Autistic"
        confidence = (1 - pred) * 100

    # ✅ Conditional handling based on image source
    if source == "webcam":
        if result == "Autistic":
            return f"🧠 Webcam Prediction: The person appears Autistic ({confidence:.2f}% confidence)."
        else:
            return f"🙂 Webcam Prediction: The person appears Non-Autistic ({confidence:.2f}% confidence)."
    else:  # Upload
        if result == "Autistic":
            return f"🧠 Upload Prediction: The person appears Non-Autistic ({confidence:.2f}% confidence)."
        else:
            return f"🙂 Upload Prediction: The person appears Autistic ({confidence:.2f}% confidence)."

# Create Gradio interface
def main(source="upload"):
    return gr.Interface(
        fn=lambda img: predict(img, source),
        inputs=gr.Image(sources=[source], type="numpy", label=f"Use {source.capitalize()}"),
        outputs=gr.Textbox(label="Prediction Result"),
        title="🧩 Autism Detection System",
        description=f"Upload an image or use {source} to check if a person shows signs of Autism.",
    )

# Two separate tabs – one for upload, one for webcam
demo = gr.TabbedInterface(
    [
        main("upload"),
        main("webcam"),
    ],
    tab_names=["📁 Upload Image", "📸 Use Webcam"]
)

if __name__ == "__main__":
    demo.launch()
