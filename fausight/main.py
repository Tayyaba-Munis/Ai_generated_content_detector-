import base64
import os
import re

import cv2
import eel
import numpy as np
import tensorflow as tf

WEB_DIR = 'web'
MODEL_PATH = 'fauxsight.keras'

eel.init(WEB_DIR)
model = tf.keras.models.load_model(MODEL_PATH)


def _load_image_from_base64(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    image_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not decode image data.')
    return image


@eel.expose
def predict_image(img_source):
    if isinstance(img_source, str) and img_source.startswith('data:'):
        img = _load_image_from_base64(img_source)
    else:
        img_path = os.path.abspath(img_source)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f'Image file not found: {img_path}')
        img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32)).astype('float32') / 255.0
    prob = float(model.predict(np.expand_dims(img, 0), verbose=0)[0][0])
    label = 'AI-Generated' if prob >= 0.5 else 'Real'
    confidence = f"{max(prob, 1 - prob) * 100:.1f}%"
    return {'label': label, 'confidence': confidence}


if __name__ == '__main__':
    eel.start('index.html', size=(1200, 800), port=8000, host='localhost')
