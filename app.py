import io, base64, time
import numpy as np
import streamlit as st
from PIL import Image
import cv2

def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_max(img, max_side=2000):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    r = max_side / s
    return cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)

def save_bytes(rgb):
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95, optimize=True)
    return buf.getvalue()

def download_button_bytes(name, bytes_):
    b64 = base64.b64encode(bytes_).decode()
    href = f'<a download="{name}" href="data:file/jpg;base64,{b64}">📥 {name} indir</a>'
    return href

def white_balance_grayworld(bgr):
    result = bgr.copy().astype(np.float32)
    avgB, avgG, avgR = np.mean(result[:,:,0]), np.mean(result[:,:,1]), np.mean(result[:,:,2])
    avg = (avgB + avgG + avgR) / 3.0 + 1e-6
    result[:,:,0] *= (avg / avgB)
    result[:,:,1] *= (avg / avgG)
    result[:,:,2] *= (avg / avgR)
    return np.clip(result, 0, 255).astype(np.uint8)

def temperature_tint(bgr, temp=0, tint=0):
    t = temp
