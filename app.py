
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
from googletrans import Translator

translator = Translator()

@st.cache_resource
def download_model():
url = 'https://drive.google.com/file/d/1Q9EHtF-qRR6fpsT-Hs08-YMXPNMLmh1A/view?usp=sharing'  
output = '/tmp/crop_disease_model.h5'
gdown.download(url, output, quiet=False)
return tf.keras.models.load_model(output)

model = download_model()
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)_Common_rust', 'Corn(maize)__Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape___Black_rot', 'Grape___Esca(Black_Measles)', 'Grape___Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
treatment_recommendations = {
'Apple___Apple_scab': 'Apply fungicides like captan or wettable sulfur (available in India); remove infected leaves.',
'Apple___Black_rot': 'Prune infected branches; apply captan fungicide during bloom.',
'Apple___Cedar_apple_rust': 'Use rust-resistant varieties; apply fungicides like triadimefon.',
'Apple___healthy': 'No action needed; maintain regular pruning and fertilization.',
'Blueberry___healthy': 'No action needed; ensure proper soil pH and irrigation.',
'Cherry(including_sour)__Powdery_mildew': 'Apply sulfur or potassium bicarbonate; improve air circulation.',
'Cherry(including_sour)__healthy': 'No action needed; maintain pruning and monitoring.',
'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides like azoxystrobin; rotate crops.',
'Corn(maize)_Common_rust': 'Use resistant varieties; apply fungicides like triazole.',
'Corn(maize)__Northern_Leaf_Blight': 'Apply fungicides like propiconazole; rotate crops.',
'Corn(maize)healthy': 'No action needed; maintain regular care.',
'Grape___Black_rot': 'Apply fungicides like captan; remove infected berries.',
'Grape___Esca(Black_Measles)': 'Prune affected vines; no effective chemical control.',
'Grape___Leaf_blight(Isariopsis_Leaf_Spot)': 'Apply fungicides like captan; remove infected leaves.',
'Grape___healthy': 'No action needed; maintain vineyard hygiene.',
'Orange___Haunglongbing(Citrus_greening)': 'Remove infected trees; control psyllid vectors with insecticides.',
'Peach___Bacterial_spot': 'Apply copper-based bactericides; prune to improve air circulation.',
'Peach___healthy': 'No action needed; maintain regular pruning and fertilization.',
'Pepper,_bell___Bacterial_spot': 'Use copper-based sprays; remove infected plant debris.',
'Pepper,_bell___healthy': 'No action needed; ensure proper irrigation and soil health.',
'Potato___Early_blight': 'Apply fungicides like chlorothalonil; rotate crops.',
'Potato___Late_blight': 'Use fungicides like mancozeb or metalaxyl (check local availability in India); destroy infected tubers.',
'Potato___healthy': 'No action needed; monitor for early disease signs.',
'Raspberry___healthy': 'No action needed; maintain pruning and soil fertility.',
'Soybean___healthy': 'No action needed; ensure proper crop rotation.',
'Squash___Powdery_mildew': 'Apply sulfur-based fungicides; improve air circulation.',
'Strawberry___Leaf_scorch': 'Remove affected leaves; apply fungicides like captan.',
'Strawberry___healthy': 'No action needed; maintain regular care.',
'Tomato___Bacterial_spot': 'Apply copper-based bactericides; avoid overhead watering.',
'Tomato___Early_blight': 'Use fungicides like chlorothalonil; remove lower infected leaves.',
'Tomato___Late_blight': 'Use fungicides like mancozeb or metalaxyl (check local availability in India); destroy infected plants.',
'Tomato___Leaf_Mold': 'Improve ventilation; apply fungicides like chlorothalonil.',
'Tomato___Septoria_leaf_spot': 'Use fungicides like mancozeb; remove infected leaves.',
'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides; increase humidity around plants.',
'Tomato___Target_Spot': 'Use fungicides like chlorothalonil; remove infected debris.',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors; remove infected plants.',
'Tomato___Tomato_mosaic_virus': 'Disinfect tools; remove infected plants.',
'Tomato___healthy': 'No action needed; maintain regular care.'
}

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Crop Disease Prediction System")
st.header("फसल रोग भविष्यवाणी प्रणाली")
language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi"])

st.sidebar.header("How to Use / उपयोग कैसे करें")
st.sidebar.text("1. Upload a clear image of a crop leaf. / फसल के पत्ते की स्पष्ट तस्वीर अपलोड करें।")
st.sidebar.text("2. Select language (English/Hindi). / भाषा चुनें (अंग्रेजी/हिंदी)।")
st.sidebar.text("3. View the predicted disease and recommendation. / भविष्यवाणी रोग और सिफारिश देखें।")
st.sidebar.text("Contact Krishi Vigyan Kendra for pesticide availability. / कीटनाशक उपलब्धता के लिए कृषि विज्ञान केंद्र से संपर्क करें।")

uploaded_file = st.file_uploader("Upload a crop image / फसल की तस्वीर अपलोड करें", type=["jpg", "png"])

if uploaded_file:

with st.spinner("Processing image... / छवि संसाधित हो रही है..."):
image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
image = cv2.resize(image, (224, 224))
image_array = np.expand_dims(image, axis=0) / 255.0
predictions = model.predict(image_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100
recommendation = treatment_recommendations.get(predicted_class, 'No recommendation available')

st.image(image, caption="Uploaded Image / अपलोड की गई तस्वीर", use_column_width=True)
if language == "Hindi":
predicted_class_hindi = translator.translate(predicted_class.replace('___', ' '), dest='hi').text
recommendation_hindi = translator.translate(recommendation, dest='hi').text
st.success(f"भविष्यवाणी रोग: {predicted_class_hindi} ({confidence:.2f}% विश्वास)")
st.info(f"सिफारिश: {recommendation_hindi}")
else:
st.success(f"Predicted Disease: {predicted_class} ({confidence:.2f}% confidence)")
st.info(f"Recommendation: {recommendation}")
