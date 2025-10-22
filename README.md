Streamlit UNet Segmentation App

This small app loads a pre-trained UNet model (place `unet_best (1).h5` in the parent folder or update the MODEL_PATH in `app.py`).

Features:
- Upload product photos (mugs, Funko Pops, perfumes).
- Run segmentation and display result overlay.
- Compute bounding box of detected product mask and display dimensions in cm and inches.

Calibration:
- The app uses a px_per_cm calibration value. You must provide the pixels-per-centimeter scale for your images to get accurate physical measurements. By default, a placeholder value is set — measure an object with known size in pixels and set the slider in the app.

Run:
- Create and activate a virtual environment.
- Install dependencies:

  pip install -r streamlit_app/requirements.txt

- Start the app:

  streamlit run streamlit_app/app.py

Notes:
- The model expects images resized to the size it was trained on; `app.py` attempts to preserve aspect ratio and resize to 256x256 by default. Adjust IMAGE_SIZE if your model requires another shape.
- If GPU acceleration is available and TensorFlow is configured, it will use it automatically.
