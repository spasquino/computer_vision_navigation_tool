import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_cnn_model
from config import DATA_DIR, MODEL_DIR, MODEL_NAME

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = build_cnn_model(input_shape=(224, 224, 3), num_classes=len(train_gen.class_indices))

    model.fit(train_gen, validation_data=val_gen, epochs=10)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

if __name__ == "__main__":
    train_model()
