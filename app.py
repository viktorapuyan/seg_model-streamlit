import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
from io import BytesIO

# Try to import tflite runtime first (smaller), fall back to tensorflow.lite if available
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    tflite_runtime_available = True
except Exception:
    TFLiteInterpreter = None
    tflite_runtime_available = False

try:
    import tensorflow as tf
    tf_available = True
except Exception:
    tf = None
    tf_available = False

# Config
# By default prefer a tflite model (same directory as H5). Update names as needed.
H5_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'unet_best (1).h5')
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'tflite_model.tflite')
IMAGE_SIZE = (512, 512)  # change if your model expects different

st.set_page_config(page_title="UNet Segmentation Demo", layout="centered")

st.title("UNet Product Segmentation")
st.write("Upload a photo of a product (mug, Funko Pop, perfume) and run segmentation.")

# Sidebar calibration
st.sidebar.header("Calibration")
px_per_cm_default = 10.0
px_per_cm = st.sidebar.slider("Pixels per centimeter (px/cm)", min_value=1.0, max_value=200.0, value=px_per_cm_default)
st.sidebar.markdown("Provide pixels-per-centimeter for your images. Measure a ruler in the image to compute this.")

def load_keras_model(path):
    if not tf_available:
        return None
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


def load_tflite_interpreter(path):
    if not os.path.exists(path):
        return None
    # prefer tflite-runtime Interpreter if available
    if tflite_runtime_available:
        interp = TFLiteInterpreter(model_path=path)
    elif tf_available:
        interp = tf.lite.Interpreter(model_path=path)
    else:
        return None
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    return interp, input_details, output_details

# Decide which model to use: prefer TFLite, fall back to H5 only if TF is available
tflite_loaded = False
keras_model = None
tflite_interp = None
tflite_input_details = None
tflite_output_details = None

if os.path.exists(TFLITE_MODEL_PATH):
    res = load_tflite_interpreter(TFLITE_MODEL_PATH)
    if res is not None:
        tflite_interp, tflite_input_details, tflite_output_details = res
        tflite_loaded = True

if not tflite_loaded:
    # try to load keras model if TF is available
    keras_model = load_keras_model(H5_MODEL_PATH)
    if keras_model is None and not tf_available:
        st.warning('Neither TFLite nor TensorFlow are available; please provide a TFLite model or enable TensorFlow in your environment.')

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def preprocess(img: Image, target_size):
    img = img.convert('RGB')
    orig_w, orig_h = img.size
    # resize keeping aspect ratio with padding
    img_resized = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized) / 255.0
    return arr.astype(np.float32), (orig_w, orig_h)


def run_keras_inference(model, image_arr):
    inp = np.expand_dims(image_arr, 0)
    pred = model.predict(inp)[0]
    # assume single-channel sigmoid output
    if pred.shape[-1] > 1:
        mask = np.argmax(pred, axis=-1)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = pred[..., 0]
        mask = (mask > 0.5).astype(np.uint8)
    return mask


def run_tflite_inference(interp, input_details, output_details, image_arr):
    # image_arr should have shape (H, W, C) and dtype float32 in [0,1] or int8 depending on the model
    inp = np.expand_dims(image_arr, 0)
    # adjust dtype if necessary
    input_index = input_details[0]['index']
    # handle quantized input
    if input_details[0].get('dtype') in (np.int8, np.uint8):
        scale, zero_point = input_details[0].get('quantization', (1.0, 0))
        inp_q = (inp / scale + zero_point).astype(input_details[0]['dtype'])
        interp.set_tensor(input_index, inp_q)
    else:
        interp.set_tensor(input_index, inp.astype(input_details[0]['dtype']))
    interp.invoke()
    out = interp.get_tensor(output_details[0]['index'])
    # if quantized output, dequantize
    if output_details[0].get('dtype') in (np.int8, np.uint8):
        scale, zero_point = output_details[0].get('quantization', (1.0, 0))
        out = (out.astype(np.float32) - zero_point) * scale
    # out shape may be (1, H, W, C) or (1, H, W)
    pred = out[0]
    if pred.ndim == 3 and pred.shape[-1] > 1:
        mask = np.argmax(pred, axis=-1)
        mask = (mask > 0).astype(np.uint8)
    else:
        if pred.ndim == 3:
            pred = pred[..., 0]
        mask = (pred > 0.5).astype(np.uint8)
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
            if (not tflite_loaded) and (keras_model is None):
                st.error("No model loaded. Place a .tflite next to the app or ensure TensorFlow is available and H5 model is present.")
            else:
                with st.spinner("Running model..."):
                    img_arr, (orig_w, orig_h) = preprocess(image, IMAGE_SIZE)
                    # choose inference backend
                    if tflite_loaded and tflite_interp is not None:
                        mask = run_tflite_inference(tflite_interp, tflite_input_details, tflite_output_details, img_arr)
                    else:
                        mask = run_keras_inference(keras_model, img_arr)
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
        if tflite_loaded:
            st.write("Model (TFLite): {}".format(TFLITE_MODEL_PATH))
        elif keras_model is not None:
            st.write("Model (Keras .h5): {}".format(H5_MODEL_PATH))
        else:
            st.write("No model loaded")

else:
    st.info("Upload an image to get started.")
