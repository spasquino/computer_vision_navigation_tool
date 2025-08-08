import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import MODEL_DIR, MODEL_NAME

def predict_image(image_path):
    model = load_model(os.path.join(MODEL_DIR, MODEL_NAME))

    image = load_img(image_path, target_size=(224, 224))
    input_arr = img_to_array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
    class_index = np.argmax(prediction)

    print(f"Predicted class index: {class_index}")
    return class_index

# Example
# predict_image("data/raw/some_folder/image.jpg")
