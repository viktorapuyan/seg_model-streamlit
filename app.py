import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
from io import BytesIO

# Config
MODEL_FILENAME = os.path.join(os.path.dirname(__file__), '..', 'tflite_model.tflite')
IMAGE_SIZE = (512, 512)  # change if your model expects different

st.set_page_config(page_title="UNet Segmentation Demo", layout="centered")

st.title("UNet Product Segmentation")
st.write("Upload a photo of a product (mug, Funko Pop, perfume) and run segmentation.")

# Sidebar calibration
st.sidebar.header("Calibration")
px_per_cm_default = 10.0
px_per_cm = st.sidebar.slider("Pixels per centimeter (px/cm)", min_value=1.0, max_value=200.0, value=px_per_cm_default)
st.sidebar.markdown("Provide pixels-per-centimeter for your images. Measure a ruler in the image to compute this.")

# Load model lazily
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.warning(f"Model file not found at {path}. Please place your model there or update MODEL_FILENAME.")
        return None
    model = tf.keras.models.load_model(path, compile=False)
    return model

model = load_model(MODEL_FILENAME)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def preprocess(img: Image, target_size):
    img = img.convert('RGB')
    orig_w, orig_h = img.size
    # resize keeping aspect ratio with padding
    img_resized = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized) / 255.0
    return arr.astype(np.float32), (orig_w, orig_h)


def run_inference(model, image_arr):
    inp = np.expand_dims(image_arr, 0)
    pred = model.predict(inp)[0]
    # assume single-channel sigmoid output
    if pred.shape[-1] > 1:
        # if multi-class, take argmax
        mask = np.argmax(pred, axis=-1)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = pred[..., 0]
        mask = (mask > 0.5).astype(np.uint8)
    return mask


def mask_to_bounding_box(mask):
    # mask: HxW binary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def draw_bbox_on_image(image: Image, bbox, color=(255, 0, 0), thickness=3):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    x, y, w, h = bbox
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, thickness)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption='Uploaded image', use_column_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Run Segmentation"):
            if model is None:
                st.error("Model not loaded. Check path in app.py and put your .h5 model there.")
            else:
                with st.spinner("Running model..."):
                    img_arr, (orig_w, orig_h) = preprocess(image, IMAGE_SIZE)
                    mask = run_inference(model, img_arr)
                    # resize mask back to original image size
                    mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    bbox = mask_to_bounding_box(mask_resized)
                    if bbox is None:
                        st.warning("No object detected in the mask.")
                    else:
                        x, y, w, h = bbox
                        # convert pixels -> cm -> inches
                        width_cm = w / px_per_cm
                        height_cm = h / px_per_cm
                        width_in = width_cm / 2.54
                        height_in = height_cm / 2.54

                        st.success("Segmentation complete")
                        st.write(f"Bounding box (pixels): x={x}, y={y}, w={w}, h={h}")
                        st.write(f"Width: {width_cm:.2f} cm ({width_in:.2f} in)")
                        st.write(f"Height: {height_cm:.2f} cm ({height_in:.2f} in)")

                        # overlay
                        bbox_image = draw_bbox_on_image(image.resize((orig_w, orig_h)), bbox)
                        st.image(bbox_image, caption='Detected bounding box', use_column_width=True)

    with col2:
        st.header("Preview & Controls")
        st.write("Calibration px/cm: ", px_per_cm)
        st.write("Image size: {} x {}".format(*image.size))
        st.write("Model file: {}".format(MODEL_FILENAME))

else:
    st.info("Upload an image to get started.")
