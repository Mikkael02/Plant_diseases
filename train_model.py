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

def train_model(train_dir, valid_dir, epochs, batch_size, model_name, learning_rate=1e-4, regularization=0.01, unfrozen_layers=30):
    img_size = (128, 128)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

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

    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )

    for layer in base_model.layers[-unfrozen_layers:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(regularization)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    total_batches = train_generator.samples // batch_size
    total_epochs = epochs
    progress_bar = st.progress(0)
    status_text = st.empty()
    epoch_text = st.empty()

    progress_callback = StreamlitProgressCallback(total_batches, total_epochs, progress_bar, status_text, epoch_text)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=[progress_callback, early_stopping]
    )

    model_path = os.path.join('model', f'{model_name}.h5')
    model.save(model_path)

    return history, model_path
