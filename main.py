import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from train_model import train_model
from sklearn.metrics import accuracy_score


# Funkcja do wczytywania dostępnych modeli
def load_models(model_dir='model/'):
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    if not model_files:  # Sprawdzenie, czy nie ma plików modeli
        st.warning("Brak modeli w folderze 'model/'. Proszę wytrenować nowy model.")
        return models

    for model_file in model_files:
        try:
            model_name = model_file.split('.')[0]  # Nazwa modelu bez rozszerzenia
            loaded_model = tf.keras.models.load_model(os.path.join(model_dir, model_file))
            loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            models[model_name] = loaded_model
        except Exception as e:
            st.error(f"Nie udało się wczytać modelu {model_file}: {e}")

    return models


# Inicjalizacja zmiennej sesji do śledzenia stanu trenowania
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

# Wczytaj dostępne modele
models = load_models()

# Lista klas (nazwy klas odpowiadające chorobom)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
               'Grape___Black_rot', 'Grape___Esca', 'Grape___healthy', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___healthy', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Corn___healthy']


# Funkcja do predykcji
def predict_image(model, image):
    img = image.resize((224, 224))  # Zmiana rozmiaru obrazu na 224x224
    img_array = np.array(img) / 255.0  # Normalizacja
    img_array = np.expand_dims(img_array, axis=0)  # Dodaj wymiar dla batch size
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]


# Funkcja do testowania modelu na obrazach w folderze testowym bez podfolderów
def test_model_on_images(model, test_dir, class_names):
    correct_predictions = 0
    total_images = 0
    predictions = []
    true_labels = []

    # Iteracja po wszystkich obrazach w katalogu test
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Rozmiar obrazu zgodny z MobileNetV2
            img_array = np.array(img) / 255.0  # Normalizacja
            img_array = np.expand_dims(img_array, axis=0)  # Dodaj wymiar dla batch size

            # Predykcja
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            # Zakładamy, że nazwa pliku ma format: 'class_xxx.jpg', gdzie 'class' to prawdziwa etykieta
            true_class_name = img_name.split('_')[0]
            true_class_index = class_names.index(true_class_name) if true_class_name in class_names else -1

            if true_class_index == -1:
                st.warning(f"Nieznana klasa dla obrazu: {img_name}")
                continue

            predictions.append(predicted_class)
            true_labels.append(true_class_index)

            total_images += 1
            if predicted_class == true_class_index:
                correct_predictions += 1

    accuracy = accuracy_score(true_labels, predictions) if total_images > 0 else 0
    return accuracy, correct_predictions, total_images


# Aplikacja Streamlit
st.title("Plant Disease Detection")

# Wysuwane menu po lewej stronie
menu_option = st.sidebar.selectbox("Wybierz opcję:",
                                   ["Strona Główna", "Wykrywanie Chorób", "Trenowanie Modelu", "Testowanie Modelu"])

if menu_option == "Strona Główna":
    st.header("Strona Główna")
    st.write("Opis projektu, autorzy i inne informacje...")

elif menu_option == "Wykrywanie Chorób":
    st.header("Wykrywanie chorób roślin 🌿")

    # Wybór modelu
    selected_model_name = st.selectbox("Wybierz model do użycia:", list(models.keys()))
    selected_model = models[selected_model_name]

    uploaded_file = st.file_uploader("Wgraj zdjęcie liścia", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Wgrane zdjęcie', use_column_width=True)

        if st.button('Przewiduj chorobę'):
            with st.spinner('Trwa przetwarzanie...'):
                result = predict_image(selected_model, image)
                st.success(f'Model przewiduje, że to: {result}')

elif menu_option == "Trenowanie Modelu":
    st.header("Trenowanie Modelu")

    # Parametry trenowania modelu
    epochs = st.number_input('Liczba epok', min_value=1, max_value=100, value=10)
    batch_size = st.number_input('Batch size', min_value=8, max_value=128, value=32)
    model_name = st.text_input('Nazwa modelu', 'nowy_model')

    # Placeholder do dynamicznej aktualizacji
    placeholder = st.empty()

    # Sprawdzaj stan trenowania
    if st.session_state.training_in_progress:
        placeholder.warning("Trenowanie jest w toku. Proszę czekać...")
    else:
        # Przywrócenie przycisku po zakończeniu trenowania z unikalnym kluczem
        if placeholder.button('Rozpocznij trenowanie', key='train_button'):
            st.session_state.training_in_progress = True  # Ustawienie stanu sesji na True
            placeholder.empty()  # Usuń przycisk, aby zapobiec ponownemu kliknięciu

            # Uruchom trenowanie modelu
            with st.spinner('Model się trenuje...'):
                train_dir = 'dataset/train'
                valid_dir = 'dataset/valid'

                # Rozpoczęcie trenowania
                history, model_path = train_model(train_dir, valid_dir, epochs, batch_size, model_name)

                # Sprawdzenie, czy trenowanie się odbyło (czy model nie istnieje)
                if history is None or model_path is None:
                    st.warning("Trenowanie zostało przerwane. Proszę wybrać inną nazwę modelu.")
                else:
                    # Po zakończeniu trenowania
                    st.success(f'Trenowanie zakończone. Model zapisano jako {model_path}')
                    models = load_models()  # Odśwież listę modeli
                    st.write(history.history)

            # Resetowanie stanu sesji po zakończeniu trenowania
            st.session_state.training_in_progress = False
            placeholder.button('Restart', key='restart_button')  # Przywróć przycisk z unikalnym kluczem

elif menu_option == "Testowanie Modelu":
    st.header("Testowanie Modelu")

    # Wybór modelu do testowania
    selected_model_name = st.selectbox("Wybierz model do testowania:", list(models.keys()))
    selected_model = models[selected_model_name]

    if st.button("Przetestuj model na zestawie testowym"):
        test_dir = "dataset/test"  # Ścieżka do folderu testowego
        accuracy, correct_predictions, total_images = test_model_on_images(selected_model, test_dir, class_names)

        st.write(f"Poprawnie sklasyfikowane obrazy: {correct_predictions}/{total_images}")
        st.write(f"Skuteczność modelu (accuracy): {accuracy * 100:.2f}%")
