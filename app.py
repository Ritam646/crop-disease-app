
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt

Initialize translator
translator = GoogleTranslator(source='auto')

Download model from Google Drive
@st.cache_resource
def download_model():
url = 'YOUR_GOOGLE_DRIVE_MODEL_LINK'  # Replace with shareable link
output = '/tmp/crop_disease_model.h5'
gdown.download(url, output, quiet=False)
return tf.keras.models.load_model(output)

Load model
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
'Potato___Late_blight': 'Use fungicides like mancozeb or metalaxyl (verify with KVK for availability in your state); destroy infected tubers.',
'Potato___healthy': 'No action needed; monitor for early disease signs.',
'Raspberry___healthy': 'No action needed; maintain pruning and soil fertility.',
'Soybean___healthy': 'No action needed; ensure proper crop rotation.',
'Squash___Powdery_mildew': 'Apply sulfur-based fungicides; improve air circulation.',
'Strawberry___Leaf_scorch': 'Remove affected leaves; apply fungicides like captan.',
'Strawberry___healthy': 'No action needed; maintain regular care.',
'Tomato___Bacterial_spot': 'Apply copper-based bactericides; avoid overhead watering.',
'Tomato___Early_blight': 'Use fungicides like chlorothalonil; remove lower infected leaves.',
'Tomato___Late_blight': 'Use fungicides like mancozeb or metalaxyl (verify with KVK for availability in your state); destroy infected plants.',
'Tomato___Leaf_Mold': 'Improve ventilation; apply fungicides like chlorothalonil.',
'Tomato___Septoria_leaf_spot': 'Use fungicides like mancozeb; remove infected leaves.',
'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides; increase humidity around plants.',
'Tomato___Target_Spot': 'Use fungicides like chlorothalonil; remove infected debris.',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors; remove infected plants.',
'Tomato___Tomato_mosaic_virus': 'Disinfect tools; remove infected plants.',
'Tomato___healthy': 'No action needed; maintain regular care.'
}

Set page config for mobile responsiveness
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Crop Disease Prediction System")
st.header("फसल रोग भविष्यवाणी प्रणाली / ফসল রোগের ভবিষ্যদ্বাণী সিস্টেম / பயிர் நோய் கணிப்பு அமைப்பு / పంట వ్యాధి అంచనా వ్యవస్థ")
language = st.selectbox("Select Language / भाषा चुनें / ভাষা নির্বাচন করুন / மொழியைத் தேர்ந்தெடு / భాషను ఎంచుకోండి", ["English", "Hindi", "Bengali", "Tamil", "Telugu"])

Sidebar with instructions and contact info
st.sidebar.header("How to Use / उपयोग कैसे करें / কীভাবে ব্যবহার করবেন / பயன்படுத்துவது எப்படி / ఎలా ఉపయోగించాలి")
st.sidebar.text("1. Upload a clear image of a crop leaf. / फसल के पत्ते की स्पष्ट तस्वीर अपलोड करें। / ফসলের পাতার একটি পরিষ্কার ছবি আপলোড করুন। / பயிர் இலையின் தெளிவான படத்தை பதிவேற்றவும். / పంట ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.")
st.sidebar.text("2. Select language (English/Hindi/Bengali/Tamil/Telugu). / भाषा चुनें (अंग्रेजी/हिंदी/बंगाली/தமிழ்/తెలుగు)। / ভাষা নির্বাচন করুন (ইংরেজি/হিন্দি/বাংলা/தமிழ்/తెలుగు)। / மொழியைத் தேர்ந்தெடு (ஆங்கிலம்/ஹிந்தி/பெங்காலி/தமிழ்/தெலுங்கு). / భాషను ఎంచుకోండి (ఆంగ్లం/హిందీ/బెంగాలీ/తమిళం/తెలుగు).")
st.sidebar.text("3. View the predicted disease and recommendation. / भविष्यवाणी रोग और सिफारिश देखें। / প্রেডিক্টেড রোগ এবং সুপারিশ দেখুন। / கணிக்கப்பட்ட நோய் மற்றும் பரிந்துரையைப் பார்க்கவும். / ఊహించిన వ్యాధి మరియు సిఫార్సును చూడండి.")
st.sidebar.text("Contact Krishi Vigyan Kendra for pesticide availability. / कीटनाशक उपलब्धता के लिए कृषि विज्ञान केंद्र से संपर्क करें। / কীটনাশকের উপলব্ধতার জন্য কৃষি বিজ্ঞান কেন্দ্রের সাথে যোগাযোগ করুন। / பூச்சிக்கொல்லி கிடைப்பதற்கு கிரிஷி விக்யான் கேந்திரத்தைத் தொடர்பு கொள்ளவும். / పురుగుమందుల లభ్యత కోసం కృషి విజ్ఞాన కేంద్రాన్ని సంప్రదించండి.")

Image uploader
uploaded_file = st.file_uploader("Upload a crop image / फसल की तस्वीर अपलोड करें / ফসলের ছবি আপলোড করুন / பயிர் படத்தை பதிவேற்றவும் / పంట చిత్రాన్ని అప్‌లోడ్ చేయండి", type=["jpg", "png"])

if uploaded_file:

Display progress spinner
with st.spinner("Processing image... / छवि संसाधित हो रही है... / ছবি প্রক্রিয়াকরণ হচ্ছে... / படம் செயலாக்கப்படுகிறது... / చిత్రం ప్రాసెస్ చేయబడుతోంది..."):
image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
image = cv2.resize(image, (224, 224))
image_array = np.expand_dims(image, axis=0) / 255.0
predictions = model.predict(image_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100
recommendation = treatment_recommendations.get(predicted_class, 'No recommendation available')

Display results
st.image(image, caption="Uploaded Image / अपलोड की गई तस्वीर / আপলোড করা ছবি / பதிவேற்றப்பட்ட படம் / అప్‌లోడ్ చేయబడిన చిత్రం", use_column_width=True)
if language == "Hindi":
predicted_class_trans = translator.translate(predicted_class.replace('', ' '), target='hi')
recommendation_trans = translator.translate(recommendation, target='hi')
st.success(f"भविष्यवाणी रोग: {predicted_class_trans} ({confidence:.2f}% विश्वास)")
st.info(f"सिफारिश: {recommendation_trans}")
elif language == "Bengali":
predicted_class_trans = translator.translate(predicted_class.replace('', ' '), target='bn')
recommendation_trans = translator.translate(recommendation, target='bn')
st.success(f"প্রেডিক্টেড রোগ: {predicted_class_trans} ({confidence:.2f}% আত্মবিশ্বাস)")
st.info(f"প্রস্তাবনা: {recommendation_trans}")
elif language == "Tamil":
predicted_class_trans = translator.translate(predicted_class.replace('', ' '), target='ta')
recommendation_trans = translator.translate(recommendation, target='ta')
st.success(f"கணிக்கப்பட்ட நோய்: {predicted_class_trans} ({confidence:.2f}% நம்பிக்கை)")
st.info(f"பரிந்துரை: {recommendation_trans}")
elif language == "Telugu":
predicted_class_trans = translator.translate(predicted_class.replace('', ' '), target='te')
recommendation_trans = translator.translate(recommendation, target='te')
st.success(f"ఊహించిన వ్యాధి: {predicted_class_trans} ({confidence:.2f}% నమ్మకం)")
st.info(f"సిఫార్సు: {recommendation_trans}")
else:
st.success(f"Predicted Disease: {predicted_class} ({confidence:.2f}% confidence)")
st.info(f"Recommendation: {recommendation}")

Confidence bar chart
st.subheader("Prediction Confidence / भविष्यवाणी विश्वास / প্রেডিকশন আত্মবিশ্বাস / கணிப்பு நம்பிக்கை / అంచనా నమ్మకం")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(class_names, predictions[0], color='green')
ax.set_ylabel("Probability")
plt.xticks(rotation=90)
st.pyplot(fig)
