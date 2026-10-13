# ScalpDetectAI

A small **Streamlit + TensorFlow image-classification prototype** intended to run inference with a fine-tuned InceptionV3 model for scalp/hair image categories.

## What is in this repository

`app.py` contains a Streamlit interface that uploads an image, resizes it to 150×150, normalizes the pixel values and calls `tf.keras.models.load_model(...).predict(...)`. The UI displays the raw prediction and maps its highest-scoring class to a hard-coded list of 11 labels.

**Important:** The required `inception_v3_finetuned_L2.h5` checkpoint is not included. The class list and preprocessing must match the model's original training configuration; that alignment cannot be independently verified from this repository alone.

## Run locally

```bash
git clone https://github.com/DevyanshMalhotra/ScalpDetectAI.git
cd ScalpDetectAI
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install streamlit tensorflow pillow numpy
```

Place your compatible `inception_v3_finetuned_L2.h5` checkpoint at the repository root, then run:

```bash
streamlit run app.py
```

The app cannot start inference without the checkpoint.

## Limitations

This is not a clinically validated diagnostic tool, and the output must not be used for medical decisions. The repository does not include training scripts, documented data provenance, validation metrics or a published model checkpoint. The existing UI is an inference demonstrator rather than a deployable diagnostic product.
