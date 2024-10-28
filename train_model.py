import os
import streamlit as st
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping
from keras.src import regularizers

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_batches, total_epochs, progress_bar, status_text, epoch_text):
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.current_batch = 0
        self.current_epoch = 0
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epoch_text = epoch_text

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_text.text(f"Epoka {self.current_epoch}/{self.total_epochs}")
        self.progress_bar.progress(0)

    def on_batch_end(self, batch, logs=None):
        self.current_batch = batch + 1
        progress = min((self.current_batch / self.total_batches), 1.0)
        self.progress_bar.progress(progress)
        status_message = f"Batch {self.current_batch}/{self.total_batches} - "
        status_message += f"Accuracy: {logs.get('accuracy', 0):.4f}, Loss: {logs.get('loss', 0):.4f}"
        self.status_text.text(status_message)

    def on_epoch_end(self, epoch, logs=None):
        self.current_batch = 0
        self.status_text.text(f"Epoka {epoch + 1} zakończona.")
        self.progress_bar.progress(1.0)

# Funkcja do trenowania modelu z MobileNetV2 i fine-tuningiem
def train_model(train_dir, valid_dir, epochs, batch_size, model_name):
    img_size = (224, 224)  # Zwiększenie rozmiaru obrazów

    # Augmentacja danych treningowych
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True
    )

    # Rescalowanie dla walidacji
    valid_datagen = ImageDataGenerator(rescale=1./255)

    # Ładowanie danych z katalogu
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Wykorzystanie pretrenowanego modelu MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Odmrożenie kilku warstw w MobileNetV2
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Dodanie własnych warstw na końcu modelu
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(38, activation='softmax')
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Tworzenie komponentów Streamlit do śledzenia postępu
    total_batches = train_generator.samples // batch_size
    total_epochs = epochs
    progress_bar = st.progress(0)
    status_text = st.empty()
    epoch_text = st.empty()

    # Callback do aktualizacji paska postępu i statusu
    progress_callback = StreamlitProgressCallback(total_batches, total_epochs, progress_bar, status_text, epoch_text)

    # Dodanie early stopping z większą wartością patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Trenowanie modelu z wyświetlaniem informacji o postępie
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=[progress_callback, early_stopping]
    )

    # Zapisanie modelu do pliku z nadaną nazwą
    model_path = os.path.join('model', f'{model_name}.h5')
    model.save(model_path)

    return history, model_path
