import os
import streamlit as st
import tensorflow as tf
import numpy as np
from train_model import train_model

def load_models(model_dir='model/'):
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    if not model_files:
        st.warning("Brak modeli w folderze 'model/'. Proszę wytrenować nowy model.")
        return models

    for model_file in model_files:
        try:
            model_name = model_file.split('.')[0]
            loaded_model = tf.keras.models.load_model(os.path.join(model_dir, model_file))
            models[model_name] = loaded_model
        except Exception as e:
            st.error(f"Nie udało się wczytać modelu {model_file}: {e}")

    return models

if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

models = load_models()

# Lista klas
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
              'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
              'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
              'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def model_prediction(model, test_image, img_size=(128, 128)):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=img_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def test_model_on_test_samples(model, test_dir='dataset/test'):
    test_labels = {
        "PotatoEarlyBlight1.JPG": "Potato___Early_blight",
        "PotatoEarlyBlight2.JPG": "Potato___Early_blight",
        "PotatoEarlyBlight3.JPG": "Potato___Early_blight",
        "PotatoEarlyBlight4.JPG": "Potato___Early_blight",
        "PotatoEarlyBlight5.JPG": "Potato___Early_blight",
        "PotatoHealthy1.JPG": "Potato___healthy",
        "PotatoHealthy2.JPG": "Potato___healthy",
        "TomatoEarlyBlight1.JPG": "Tomato___Early_blight",
        "TomatoEarlyBlight2.JPG": "Tomato___Early_blight",
        "TomatoEarlyBlight3.JPG": "Tomato___Early_blight",
        "TomatoEarlyBlight4.JPG": "Tomato___Early_blight",
        "TomatoEarlyBlight5.JPG": "Tomato___Early_blight",
        "TomatoEarlyBlight6.JPG": "Tomato___Early_blight",
        "TomatoHealthy1.JPG": "Tomato___healthy",
        "TomatoHealthy2.JPG": "Tomato___healthy",
        "TomatoHealthy3.JPG": "Tomato___healthy",
        "TomatoHealthy4.JPG": "Tomato___healthy",
        "TomatoYellowCurlVirus1.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "TomatoYellowCurlVirus2.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "TomatoYellowCurlVirus3.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "TomatoYellowCurlVirus4.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "TomatoYellowCurlVirus5.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "TomatoYellowCurlVirus6.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "AppleCedarRust1.JPG": "Apple___Cedar_apple_rust",
        "AppleCedarRust2.JPG": "Apple___Cedar_apple_rust",
        "AppleCedarRust3.JPG": "Apple___Cedar_apple_rust",
        "AppleCedarRust4.JPG": "Apple___Cedar_apple_rust",
        "AppleScab1.JPG": "Apple___Apple_scab",
        "AppleScab2.JPG": "Apple___Apple_scab",
        "AppleScab3.JPG": "Apple___Apple_scab",
        "CornCommonRust1.JPG": "Corn_(maize)___Common_rust_",
        "CornCommonRust2.JPG": "Corn_(maize)___Common_rust_",
        "CornCommonRust3.JPG": "Corn_(maize)___Common_rust_"
    }

    results = []
    correct_predictions = 0

    for file_name in os.listdir(test_dir):
        if file_name in test_labels:
            file_path = os.path.join(test_dir, file_name)
            actual_class = test_labels[file_name]
            result_index = model_prediction(model, file_path)
            predicted_class = class_names[result_index] if result_index < len(class_names) else "Nieznana klasa"

            is_correct = (predicted_class == actual_class)
            if is_correct:
                correct_predictions += 1
            results.append({
                'Nazwa pliku': file_name,
                'Klasa rzeczywista': actual_class,
                'Przewidywana klasa': predicted_class,
                'Poprawnie': is_correct
            })

    accuracy = (correct_predictions / len(test_labels)) * 100
    st.write(f"Dokładność modelu na zdjęciach testowych: {accuracy:.2f}%")
    st.table(results)

    return results

# Aplikacja Streamlit
st.title("Aplikacja do wykrywania chorób roślin 🌿")

def show_footer():
    st.sidebar.markdown('---')
    st.sidebar.markdown("""
        ### **Autorzy projektu 👨‍💻:**   
        Michał Pękowski - 4ID15B    
        Dawid Rupniewski - 4ID15B   
        Jakub Stalewski  
    """)

st.sidebar.title("Menu")
menu_option = st.sidebar.selectbox("Wybierz opcję:", ["Strona Główna", "Wykrywanie Chorób", "Trenowanie Modelu",
                                                      "Testowanie Modelu"])

if menu_option == "Strona Główna":
    st.markdown("""
    ### Witaj w Systemie Rozpoznawania Chorób Roślin! 🌱🔍

    Projekt został stworzony, aby ułatwić identyfikację chorób roślin na podstawie zdjęć ich liści.

    ### Jak Działa Aplikacja?
    System oferuje trzy główne funkcje:
    1. **Wykrywanie Chorób** – Prześlij zdjęcie liścia, a model przeanalizuje je, aby wykryć ewentualne oznaki choroby.
    2. **Trenowanie Nowych Modeli** – Możliwość trenowania własnych modeli z odpowiednimi parametrami.
    3. **Testowanie Modeli** – Sprawdzenie dokładności modeli na zbiorze testowym.

    ### Wykorzystane Technologie
    - **TensorFlow i Keras**: Nasze modele opierają się na bibliotekach deep learning, które pozwalają na skuteczne przetwarzanie obrazów.
    - **Streamlit**: Framework użyty do budowy interfejsu użytkownika.
    - **MobilNetV2 z Fine-Tuningiem**: Model, który doskonale sprawdza się w zadaniach klasyfikacji obrazów o wysokiej jakości, przy stosunkowo niskim zapotrzebowaniu na zasoby obliczeniowe.

    ### Szczegóły Zbioru Danych 🌱
    Używamy zbioru danych, który obejmuje obrazy zdrowych i chorych liści różnych roślin:
    - **Zbiór treningowy**: 70295 obrazów
    - **Zbiór walidacyjny**: 17572 obrazów
    - **Zbiór testowy**: 33 obrazy do walidacji końcowej.

    Zbiór ten zawiera 38 klas, m.in. dla chorób jabłoni, winorośli, pomidorów i ziemniaków. Próbki te zostały poddane augmentacji, aby zwiększyć różnorodność i poprawić wydajność modelu.
    Możemy go znaleźć na stronie kaggle.com [Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/)

    ### Dobór Modelu i Parametrów
    W trakcie projektowania systemu:
    - **Augmentacja danych**: Obrót, przesunięcia, skalowanie obrazów, aby zwiększyć różnorodność danych.
    - **Regularizacja**: Dodanie dropout i L2 regularizacji, aby zmniejszyć ryzyko nadmiernego dopasowania.
    - **Fine-tuning**: Zastosowanie modyfikacji końcowych warstw sieci, co pozwala na dopasowanie do specyficznych cech zbioru.

    ### Interfejs Użytkownika 🌐
    1. **Wykrywanie Chorób**: Załaduj zdjęcie liścia i wybierz model do predykcji.
    2. **Trenowanie Modelu**: Wybierz parametry, takie jak liczba epok, batch size, i rozpocznij trenowanie nowego modelu.
    3. **Testowanie Modelu**: Funkcja pozwala na ocenę skuteczności modelu na zbiorze testowym.

    ### Jak Korzystać z Aplikacji? 📸
    1. Wybierz **Wykrywanie Chorób** i załaduj zdjęcie.
    2. Kliknij "Przewiduj chorobę", aby uzyskać wynik i diagnozę.
    """)

elif menu_option == "Wykrywanie Chorób":
    st.header("Wykrywanie chorób roślin")

    disease_info = {
        "Apple___Apple_scab": {
            "description": "Jabłoń - Parch jabłoni (Venturia inaequalis). Powoduje czarne plamy na liściach, pędy i owoce.",
            "suggestions": "Usuń porażone liście i owoce, stosuj fungicydy."
        },
        "Apple___Black_rot": {
            "description": "Jabłoń - Czarna zgnilizna (Botryosphaeria obtusa). Prowadzi do czernienia liści i gnicia owoców.",
            "suggestions": "Wytnij i usuń zainfekowane części drzewa. Użyj odpowiednich fungicydów."
        },
        "Apple___Cedar_apple_rust": {
            "description": "Jabłoń - Rdza jabłoniowa (Gymnosporangium juniperi-virginianae). Powoduje pomarańczowe plamy na liściach.",
            "suggestions": "Usuń porażone liście, stosuj fungicydy w razie potrzeby."
        },
        "Apple___healthy": {
            "description": "Jabłoń - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Blueberry___healthy": {
            "description": "Borówka - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Cherry_(including_sour)___Powdery_mildew": {
            "description": "Wiśnia/Czereśnia - Mączniak prawdziwy (Podosphaera clandestina). Pojawiają się białe naloty na liściach.",
            "suggestions": "Usuń porażone liście, stosuj odpowiednie fungicydy."
        },
        "Cherry_(including_sour)___healthy": {
            "description": "Wiśnia/Czereśnia - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
            "description": "Kukurydza - Szara plamistość liści (Cercospora zeae-maydis). Powoduje ciemne plamy na liściach.",
            "suggestions": "Zastosuj fungicydy i usuń porażone liście."
        },
        "Corn_(maize)___Common_rust_": {
            "description": "Kukurydza - Rdza kukurydzy (Puccinia sorghi). Czerwone plamy na liściach.",
            "suggestions": "Stosuj fungicydy i usuwaj porażone rośliny."
        },
        "Corn_(maize)___Northern_Leaf_Blight": {
            "description": "Kukurydza - Północna plamistość liści (Setosphaeria turcica). Pojawiają się wydłużone, szare plamy.",
            "suggestions": "Stosuj odmiany odporne i używaj fungicydów."
        },
        "Corn_(maize)___healthy": {
            "description": "Kukurydza - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Grape___Black_rot": {
            "description": "Winorośl - Czarna zgnilizna (Guignardia bidwellii). Powoduje czernienie owoców i liści.",
            "suggestions": "Usuń zainfekowane owoce i liście, stosuj fungicydy."
        },
        "Grape___Esca_(Black_Measles)": {
            "description": "Winorośl - Apopleksja (Phaeomoniella chlamydospora). Plamy na liściach, powoduje zamieranie roślin.",
            "suggestions": "Stosuj odpowiednie fungicydy i wytnij chore części roślin."
        },
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
            "description": "Winorośl - Plamistość liści (Isariopsis clavispora). Powoduje brunatne plamy na liściach.",
            "suggestions": "Usuń porażone liście, stosuj fungicydy."
        },
        "Grape___healthy": {
            "description": "Winorośl - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Orange___Haunglongbing_(Citrus_greening)": {
            "description": "Pomarańcza - Zielona choroba cytrusów (Candidatus Liberibacter spp.). Powoduje żółknięcie liści i zniekształcenie owoców.",
            "suggestions": "Usuń chore drzewa i monitoruj szkodniki przenoszące chorobę."
        },
        "Peach___Bacterial_spot": {
            "description": "Brzoskwinia - Bakterioza (Xanthomonas campestris). Powoduje plamy na liściach i owocach.",
            "suggestions": "Usuń zainfekowane części i stosuj bakteriofagi lub miedziowe środki ochrony."
        },
        "Peach___healthy": {
            "description": "Brzoskwinia - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Pepper,_bell___Bacterial_spot": {
            "description": "Papryka - Bakterioza (Xanthomonas campestris). Plamy na liściach i owocach.",
            "suggestions": "Stosuj bakteriofagi lub miedziowe środki ochrony."
        },
        "Pepper,_bell___healthy": {
            "description": "Papryka - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Potato___Early_blight": {
            "description": "Ziemniak - Zaraza wczesna (Alternaria solani). Powoduje brązowe plamy na liściach.",
            "suggestions": "Stosuj fungicydy na bazie miedzi, usuń porażone liście."
        },
        "Potato___Late_blight": {
            "description": "Ziemniak - Zaraza późna (Phytophthora infestans). Powoduje wodniste plamy na liściach i gnijące bulwy.",
            "suggestions": "Usuń porażone rośliny, stosuj fungicydy."
        },
        "Potato___healthy": {
            "description": "Ziemniak - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Raspberry___healthy": {
            "description": "Malina - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Soybean___healthy": {
            "description": "Soja - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Squash___Powdery_mildew": {
            "description": "Dynia - Mączniak prawdziwy (Erysiphe cichoracearum). Pojawia się biały nalot na liściach.",
            "suggestions": "Stosuj fungicydy i usuń porażone liście."
        },
        "Strawberry___Leaf_scorch": {
            "description": "Truskawka - Plamistość liści (Diplocarpon earlianum). Pojawiają się brązowe plamy na liściach.",
            "suggestions": "Usuń porażone liście i stosuj fungicydy."
        },
        "Strawberry___healthy": {
            "description": "Truskawka - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        },
        "Tomato___Bacterial_spot": {
            "description": "Pomidor - Bakterioza (Xanthomonas campestris). Pojawiają się ciemne plamy na liściach i owocach.",
            "suggestions": "Stosuj środki ochrony miedziowej i bakteriofagi."
        },
        "Tomato___Early_blight": {
            "description": "Pomidor - Zaraza wczesna (Alternaria solani). Powoduje brązowe plamy na liściach.",
            "suggestions": "Usuń porażone liście i stosuj fungicydy."
        },
        "Tomato___Late_blight": {
            "description": "Pomidor - Zaraza późna (Phytophthora infestans). Powoduje wodniste plamy na liściach i gnijące owoce.",
            "suggestions": "Stosuj fungicydy, usuń porażone rośliny."
        },
        "Tomato___Leaf_Mold": {
            "description": "Pomidor - Pleśń liściowa (Passalora fulva). Pojawia się szary nalot na liściach.",
            "suggestions": "Stosuj odpowiednie fungicydy i zapewnij dobrą wentylację."
        },
        "Tomato___Septoria_leaf_spot": {
            "description": "Pomidor - Septorioza liści (Septoria lycopersici). Powoduje brązowe plamy na liściach.",
            "suggestions": "Usuń porażone liście i stosuj fungicydy."
        },
        "Tomato___Spider_mites Two-spotted_spider_mite": {
            "description": "Pomidor - Przędziorek (Tetranychus urticae). Powoduje żółte plamy na liściach.",
            "suggestions": "Stosuj środki owadobójcze i utrzymuj wilgotność."
        },
        "Tomato___Target_Spot": {
            "description": "Pomidor - Okrągła plamistość (Corynespora cassiicola). Okrągłe plamy na liściach i owocach.",
            "suggestions": "Stosuj fungicydy, usuń porażone części rośliny."
        },
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
            "description": "Pomidor - Żółta mozaika liściowa wirusa TYLCV. Powoduje żółknięcie liści i deformacje.",
            "suggestions": "Stosuj środki owadobójcze, aby kontrolować wektory wirusa."
        },
        "Tomato___Tomato_mosaic_virus": {
            "description": "Pomidor - Wirus mozaiki pomidora (ToMV). Powoduje mozaikowate plamy na liściach.",
            "suggestions": "Stosuj odpowiednie środki przeciwwirusowe i usuń porażone rośliny."
        },
        "Tomato___healthy": {
            "description": "Pomidor - Zdrowa roślina. Brak oznak choroby.",
            "suggestions": "Kontynuuj odpowiednią pielęgnację."
        }
    }

    if models:
        selected_model_name = st.selectbox("Wybierz model do użycia:", list(models.keys()))
        if selected_model_name:
            selected_model = models[selected_model_name]
    else:
        st.warning("Brak dostępnych modeli.")

    uploaded_file = st.file_uploader("Wgraj zdjęcie liścia", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None and 'selected_model' in locals():
        st.image(uploaded_file, caption='Wgrane zdjęcie', use_column_width=True)
        if st.button('Przewiduj chorobę'):
            with st.spinner('Trwa przetwarzanie...'):
                result_index = model_prediction(selected_model, uploaded_file)
                result_class = class_names[result_index] if result_index < len(class_names) else "Nieznana klasa"

                if result_class in disease_info:
                    description = disease_info[result_class]["description"]
                    suggestions = disease_info[result_class]["suggestions"]
                    st.success(f'Model przewiduje, że to: {result_class}')
                    st.write(f"**Opis:** {description}")
                    st.write(f"**Wskazówki:** {suggestions}")
                else:
                    st.success(f'Model przewiduje, że to: {result_class}')
                    st.write("Brak dodatkowych informacji dla tej klasy.")

elif menu_option == "Trenowanie Modelu":
    st.header("Trenowanie Modelu")

    # Ustawienia trenowania
    epochs = st.number_input('Liczba epok', min_value=1, max_value=100, value=10)
    batch_size = st.number_input('Batch size', min_value=8, max_value=128, value=32)
    model_name = st.text_input('Nazwa modelu', 'nowy_model')

    # Dodatkowe parametry trenowania
    st.subheader("Dodatkowe ustawienia")
    learning_rate = st.slider("Learning Rate (współczynnik uczenia)", min_value=1e-5, max_value=1e-2, value=1e-4, step=1e-5)
    regularization = st.slider("Regularization (L2)", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
    unfrozen_layers = st.number_input("Liczba warstw do odmrożenia (fine-tuning)", min_value=0, max_value=100, value=30)

    placeholder = st.empty()

    if st.session_state.training_in_progress:
        placeholder.warning("Trenowanie jest w toku.")
    else:
        if placeholder.button('Rozpocznij trenowanie', key='train_button'):
            st.session_state.training_in_progress = True
            placeholder.empty()

            with st.spinner('Model się trenuje...'):
                train_dir = 'dataset/train'
                valid_dir = 'dataset/valid'
                history, model_path = train_model(
                    train_dir=train_dir,
                    valid_dir=valid_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    model_name=model_name,
                    learning_rate=learning_rate,
                    regularization=regularization,
                    unfrozen_layers=unfrozen_layers
                )
                if history and model_path:
                    st.success(f'Trenowanie zakończone. Model zapisano jako {model_path}')
                    models = load_models()
                    st.write(history.history)

            st.session_state.training_in_progress = False
            placeholder.button('Restart', key='restart_button')

elif menu_option == "Testowanie Modelu":
    st.header("Testowanie Modelu")

    if models:
        selected_model_name = st.selectbox("Wybierz model do testu:", list(models.keys()))
        if selected_model_name:
            selected_model = models[selected_model_name]
            results = test_model_on_test_samples(selected_model)
            st.write("Wyniki testu na obrazach testowych:")
            st.table(results)
    else:
        st.warning("Brak dostępnych modeli.")
#Autoirzy
show_footer()