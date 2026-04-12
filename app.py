import os
import cv2
import hashlib
import shutil
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from datetime import datetime
import warnings

# Suppress Starlette and Gradio warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

# Configure GPU memory growth dynamically
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Configure Matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup samples directory
os.makedirs("samples", exist_ok=True)

# 1. Model Configuration (12 Models)
MODEL_PATHS = {
    "Xception (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E27_CLAHE_Contrast_Xception_P4I3A3M7T4.keras",

    "Xception (Contrast, Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E23_CLAHE_Contrast_Xception_P4I3A2M7T4.keras",

    "ImprovedCNN (Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E5_ImprovedCNN_P1I1A3M2T2.h5",

    "MobileNetV2 (Base)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E9_MobileNetV2_P1I3A2M3T3.h5",

    "ResNet50 (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E26_CLAHE_Contrast_ResNet50_P4I3A3M6T4.keras",

    "Xception (Base)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E21_Xception_P1I3A2M7T3.keras",

    "ImprovedCNN (Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E4_ImprovedCNN_P1I1A2M2T1.h5",

    "ImprovedCNN (224 Gray)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E8_CLAHE_Contrast_224Gray_ImprovedCNN_P4I2A2M2T2.h5",

    "MobileNetV2 (CLAHE)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E10_CLAHE_MobileNetV2_P2I3A2M3T3.h5",

    "ResNet50 (Contrast, Light)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E20_CLAHE_Contrast_ResNet50_P4I3A2M6T4.keras",

    "ImprovedCNN (Contrast)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E7_CLAHE_Contrast_ImprovedCNN_P4I1A2M2T2.h5",

    "DenseNet121 (Contrast, Medium)":
        r"C:\CODES__SSD512\Pneumonia Project\Models\E25_CLAHE_Contrast_DenseNet121_P4I3A3M5T4.keras"
}

loaded_models = {}

# DepthwiseConv2D compatibility layer for TF 2.10
class LegacyDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

print("Initializing models...")
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            print(f"Loading actual model {name} from {path}...")
            # Pass custom class for backward compatibility
            loaded_models[name] = tf.keras.models.load_model(
                path, 
                custom_objects={'DepthwiseConv2D': LegacyDepthwiseConv2D},
                compile=False
            )
            print(f"Successfully loaded actual model {name}.")
        except Exception as e:
            print(f"Failed to load model {name} (Error: {e}). Running in Simulation Mode.")
            loaded_models[name] = None
    else:
        print(f"Model path {path} not found. Running in Simulation Mode.")
        loaded_models[name] = None