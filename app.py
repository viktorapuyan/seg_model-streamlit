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

st.set_page_config(page_title="Product Measurement Demo", layout="centered")

st.title("UNet Product Segmentation")
st.write("Upload a photo of a product (mug, Funko Pop, perfume) and run segmentation.")

# We'll compute px_per_cm automatically from image metadata when possible.
def _rational_to_float(r):
    try:
        if isinstance(r, tuple) and len(r) == 2:
            return float(r[0]) / float(r[1]) if r[1] != 0 else float(r[0])
        return float(r)
    except Exception:
        return None


def get_px_per_cm(pil_img):
    """Attempt to derive pixels-per-centimeter from image metadata (dpi or EXIF). Fallback to 96 DPI.

    Returns px_per_cm (float).
    """
    # 1 inch = 2.54 cm
    DEFAULT_DPI = 96.0
    # check info['dpi'] first (PIL sets this for some formats)
    try:
        info = getattr(pil_img, 'info', {}) or {}
        if 'dpi' in info:
            dpi = info['dpi'][0]
            if dpi and dpi > 0:
                return dpi / 2.54
    except Exception:
        pass

    # try EXIF tags for XResolution/YResolution and ResolutionUnit
    try:
        exif = pil_img._getexif() if hasattr(pil_img, '_getexif') else None
        if exif:
            # EXIF tag ids: 282 = XResolution, 283 = YResolution, 296 = ResolutionUnit
            xres = exif.get(282) or exif.get(283)
            unit = exif.get(296)
            xval = _rational_to_float(xres) if xres is not None else None
            if xval is not None and xval > 0:
                # ResolutionUnit: 2 means inches, 3 means cm
                if unit == 3:
                    # xval is pixels per cm
                    return xval
                else:
                    # assume inches
                    return xval / 2.54
    except Exception:
        pass

    # fallback to default dpi
    return DEFAULT_DPI / 2.54

def load_keras_model(path):
    if not tf_available:
        return None
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


def load_tflite_interpreter(path):
    # Return interpreter tuple or raise exception to be handled by caller
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found at {path}")
    try:
        # prefer tflite-runtime Interpreter if available
        if tflite_runtime_available:
            interp = TFLiteInterpreter(model_path=path)
        elif tf_available:
            interp = tf.lite.Interpreter(model_path=path)
        else:
            raise RuntimeError('No tflite runtime or TensorFlow available to load TFLite model')
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        return interp, input_details, output_details
    except Exception as e:
        # Re-raise so caller can report the reason
        raise

# Decide which model to use: prefer TFLite, fall back to H5 only if TF is available
tflite_loaded = False
keras_model = None
tflite_interp = None
tflite_input_details = None
tflite_output_details = None

# Allow uploading a .tflite model via the UI or search common locations
st.sidebar.header("Model")
uploaded_tflite = st.sidebar.file_uploader("Upload a .tflite model (optional)", type=["tflite"])
candidate_paths = [TFLITE_MODEL_PATH, os.path.join(os.path.dirname(__file__), 'model.tflite'), os.path.join(os.path.dirname(__file__), '..', 'compressed_models', 'unet_best.dynamic.tflite')]

if uploaded_tflite is not None:
    # save uploaded model to the app directory so interpreter can load from filesystem
    saved_path = os.path.join(os.path.dirname(__file__), os.path.basename(uploaded_tflite.name))
    with open(saved_path, 'wb') as f:
        f.write(uploaded_tflite.read())
    candidate_paths.insert(0, saved_path)

# find first loadable tflite and collect diagnostics
load_errors = []
for p in candidate_paths:
    exists = os.path.exists(p)
    if not exists:
        load_errors.append((p, 'missing'))
        continue
    try:
        res = load_tflite_interpreter(p)
        tflite_interp, tflite_input_details, tflite_output_details = res
        tflite_loaded = True
        TFLITE_MODEL_PATH = p
        load_errors.append((p, 'loaded'))
        break
    except Exception as e:
        load_errors.append((p, f'error: {e}'))

if not tflite_loaded:
    # try to load keras model if TF is available
    keras_model = load_keras_model(H5_MODEL_PATH)
    if keras_model is None and not tf_available:
        st.warning('Neither TFLite model nor TensorFlow are available; upload a .tflite model or enable TensorFlow in your environment.')

# Show diagnostic summary of candidate model paths in sidebar
st.sidebar.markdown("**Model load diagnostics**")
for p, status in load_errors:
    st.sidebar.text(f"{p} -> {status}")
if uploaded_tflite is not None:
    st.sidebar.success(f"Uploaded model saved to: {saved_path}")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def preprocess(img: Image, target_size):
    img = img.convert('RGB')
    orig_w, orig_h = img.size
    # resize to model input size (no padding here for simplicity)
    img_resized = img.resize(target_size, Image.BILINEAR)
    # return image in 0-255 as in your notebook pipeline
    arr = np.array(img_resized).astype(np.float32)
    return arr, (orig_w, orig_h)


def read_image_mask_from_pil(pil_img, mask=False, size=IMAGE_SIZE):
    """Mimic the notebook's read_image_mask: return numpy arrays.
    If mask True, returns binary mask (0/1) of shape HxW or HxW x1.
    """
    if mask:
        img = pil_img.convert('L').resize(size, Image.NEAREST)
        arr = np.array(img)
        # threshold to binary
        arr = (arr > 127).astype(np.uint8)
        return arr
    else:
        img = pil_img.convert('RGB').resize(size, Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        return arr


def preprocess_image(path):
    # path can be a PIL Image or a path-like; accept PIL Image for uploaded content
    if isinstance(path, Image.Image):
        img = read_image_mask_from_pil(path, mask=False)
    else:
        img_pil = Image.open(path).convert('RGB')
        img = read_image_mask_from_pil(img_pil, mask=False)
    # keep 0-255 as notebook did
    return img


def preprocess_mask(path):
    if isinstance(path, Image.Image):
        m = read_image_mask_from_pil(path, mask=True)
    else:
        m_pil = Image.open(path).convert('L')
        m = read_image_mask_from_pil(m_pil, mask=True)
    return m


def run_keras_inference(model, image_arr):
    inp = np.expand_dims(image_arr, 0)
    pred = model.predict(inp)[0]
    # handle multi-class softmax (channel 1 = foreground) or single-channel sigmoid
    if pred.ndim == 3 and pred.shape[-1] > 1:
        # use second channel as foreground probability
        fg = pred[..., 1]
        mask = (fg > 0.5).astype(np.uint8)
    else:
        # single channel
        prob = pred[..., 0] if pred.ndim == 3 else pred
        mask = (prob > 0.5).astype(np.uint8)
    return mask


def run_tflite_inference(interp, input_details, output_details, image_arr):
    # image_arr should have shape (H, W, C) and dtype float32 in [0,1] or int8 depending on the model
    inp = np.expand_dims(image_arr, 0)
    # adjust dtype if necessary
    input_index = input_details[0]['index']
    # handle quantized input
    if input_details[0].get('dtype') in (np.int8, np.uint8):
        scale, zero_point = input_details[0].get('quantization', (1.0, 0))
        # if model expects quantized input, assume original image_arr in 0-255 and convert
        inp_q = ((inp) / scale + zero_point).astype(input_details[0]['dtype'])
        interp.set_tensor(input_index, inp_q)
    else:
        # if model expects float input, set directly. Many TF models accept 0-255 inputs if trained that way.
        interp.set_tensor(input_index, inp.astype(input_details[0]['dtype']))
    interp.invoke()
    out = interp.get_tensor(output_details[0]['index'])
    # if quantized output, dequantize
    if output_details[0].get('dtype') in (np.int8, np.uint8):
        scale, zero_point = output_details[0].get('quantization', (1.0, 0))
        out = (out.astype(np.float32) - zero_point) * scale
    # out shape may be (1, H, W, C) or (1, H, W)
    pred = out[0]
    # handle multi-channel softmax like Keras model (use channel 1 as foreground)
    if pred.ndim == 3 and pred.shape[-1] > 1:
        fg = pred[..., 1]
        mask = (fg > 0.5).astype(np.uint8)
    else:
        if pred.ndim == 3:
            prob = pred[..., 0]
        else:
            prob = pred
        mask = (prob > 0.5).astype(np.uint8)
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
    image = Image.open(uploaded).convert('RGB')

    # Run segmentation immediately
    if (not tflite_loaded) and (keras_model is None):
        st.error("No model loaded. Place a .tflite next to the app or ensure TensorFlow is available and H5 model is present.")
    else:
        with st.spinner("Running segmentation..."):
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
            st.image(image, caption='Uploaded image', use_column_width=True)
        else:
            x, y, w, h = bbox
            # compute pixels-per-cm automatically from image metadata, then compute dimensions
            px_per_cm = get_px_per_cm(image)
            width_cm = w / px_per_cm
            height_cm = h / px_per_cm
            width_in = width_cm / 2.54
            height_in = height_cm / 2.54

            # draw bbox (green) on original image and show
            bbox_image = draw_bbox_on_image(image, bbox, color=(0, 255, 0), thickness=4)

            st.markdown("### Original Image")
            st.image(bbox_image, use_column_width=False)

            # show measurements in two columns similar to the example
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(label="Width", value=f"{width_in:.2f} in", delta=f"{width_cm:.2f} cm")
            with mcol2:
                st.metric(label="Height", value=f"{height_in:.2f} in", delta=f"{height_cm:.2f} cm")

            st.write(f"Bounding box (pixels): x={x}, y={y}, w={w}, h={h}")
            st.success("Segmentation complete")

else:
    st.info("Upload an image to get started.")
