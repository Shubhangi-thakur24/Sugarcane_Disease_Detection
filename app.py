
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from io import BytesIO
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import random
import cv2
from PIL import Image

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Sugarcane Leaf Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TRANSLATION DICTIONARY FOR MULTI-LANGUAGE SUPPORT
TRANSLATIONS = {
    "English": {
        "nav_dashboard": "Dashboard",
        "nav_detection": "Disease Detection",
        "nav_analytics": "Analytics",
        "nav_feedback": "Feedback",
        "nav_settings": "Settings",
        "nav_header": "Navigation",
        "quick_stats": "Quick Stats",
        "total_scans": "Total Scans",
        "healthy_leaves": "Healthy Leaves",
        "diseased_leaves": "Diseased Leaves",
        "logout": "Logout",
        "login": "Login",
        "signup": "Create Account",
        "email": "Email Address",
        "password": "Password",
        "name": "Full Name",
        "confirm_password": "Confirm Password",
        "register": "Register",
        "back_to_login": "Back to Login",
        "sys_dashboard": "System Dashboard",
        "active_days": "Active Days",
        "recent_history": "Recent Scan History",
        "no_scans": "No scans available. Go to 'Disease Detection' to perform your first scan.",
        "disease_detection": "Disease Detection",
        "upload_instruction": "Upload or capture a sugarcane leaf image for prediction",
        "choose_image": "Choose Leaf Image",
        "take_photo": "Take Photo",
        "uploaded_image": "Uploaded Image",
        "analysis_result": "Analysis Result",
        "processing": "Processing...",
        "predicted_disease": "Predicted Disease",
        "confidence": "Confidence",
        "model_confidence": "Model Confidence",
        "high_conf": "High model confidence",
        "mod_conf": "Moderate model confidence",
        "low_conf": "Low model confidence - please verify with more images.",
        "inference_time": "Inference Time",
        "disease_info": "Disease Information",
        "gradcam_heatmap": "Grad-CAM Heatmap (Model Focus Regions)",
        "gradcam_caption": "Regions most influential for the prediction",
        "gradcam_error": "Grad-CAM visualization not available for this model configuration.",
        "analytics_title": "Analytics Dashboard",
        "disease_dist": "Disease Distribution (Donut)",
        "conf_dist": "Confidence Distribution (Bar)",
        "monthly_trend": "Monthly Scan Trend (Area)",
        "community_fb": "User Feedback Board",
        "submit_fb": "Share Your Feedback",
        "your_name": "Your Name",
        "location": "Location",
        "rating": "Rating",
        "feedback": "Feedback",
        "submit": "Submit",
        "preferences": "Preferences",
        "theme": "Theme",
        "language": "Language",
        "data_management": "Data Management",
        "clear_history": "Clear Scan History",
        "save_settings": "Save Settings",
        "settings_saved": "Settings saved successfully!",
        "footer_text": "Sugarcane Leaf Disease Detector | Powered by Deep Learning",
        "footer_sub": "Designed for smarter agricultural decision-making"
    },
    "Hindi": {
        "nav_dashboard": "डैशबोर्ड",
        "nav_detection": "रोग की पहचान",
        "nav_analytics": "विश्लेषण",
        "nav_feedback": "फीडबैक",
        "nav_settings": "सेटिंग्स",
        "nav_header": "नेविगेशन",
        "quick_stats": "त्वरित आँकड़े",
        "total_scans": "कुल स्कैन",
        "healthy_leaves": "स्वस्थ पत्तियां",
        "diseased_leaves": "रोगग्रस्त पत्तियां",
        "logout": "लॉगआउट",
        "login": "लॉगिन",
        "signup": "खाता बनाएं",
        "email": "ईमेल पता",
        "password": "पासवर्ड",
        "name": "पूरा नाम",
        "confirm_password": "पासवर्ड की पुष्टि करें",
        "register": "पंजीकरण करें",
        "back_to_login": "लॉगिन पर वापस जाएं",
        "sys_dashboard": "सिस्टम डैशबोर्ड",
        "active_days": "सक्रिय दिन",
        "recent_history": "हाल का स्कैन इतिहास",
        "no_scans": "कोई स्कैन उपलब्ध नहीं है। अपना पहला स्कैन करने के लिए 'रोग की पहचान' पर जाएं।",
        "disease_detection": "रोग की पहचान",
        "upload_instruction": "पूर्वानुमान के लिए गन्ने की पत्ती की छवि अपलोड या कैप्चर करें",
        "choose_image": "पत्ती की छवि चुनें",
        "take_photo": "फोटो खींचें",
        "uploaded_image": "अपलोड की गई छवि",
        "analysis_result": "विश्लेषण परिणाम",
        "processing": "प्रसंस्करण हो रहा है...",
        "predicted_disease": "अनुमानित रोग",
        "confidence": "सटीकता",
        "model_confidence": "मॉडल सटीकता",
        "high_conf": "उच्च मॉडल सटीकता",
        "mod_conf": "मध्यम मॉडल सटीकता",
        "low_conf": "कम मॉडल सटीकता - कृपया अधिक छवियों के साथ सत्यापित करें।",
        "inference_time": "अनुमान समय",
        "disease_info": "रोग की जानकारी",
        "gradcam_heatmap": "Grad-CAM हीटमैप (मॉडल फोकस क्षेत्र)",
        "gradcam_caption": "पूर्वानुमान के लिए सबसे प्रभावशाली क्षेत्र",
        "gradcam_error": "इस मॉडल कॉन्फ़िगरेशन के लिए Grad-CAM विज़ुअलाइज़ेशन उपलब्ध नहीं है।",
        "analytics_title": "विश्लेषण डैशबोर्ड",
        "disease_dist": "रोग वितरण (डोनट)",
        "conf_dist": "सटीकता वितरण",
        "monthly_trend": "मासिक स्कैन प्रवृत्ति क्षेत्र ग्राफ",
        "community_fb": "उपयोगकर्ता फीडबैक बोर्ड",
        "submit_fb": "अपनी फीडबैक साझा करें",
        "your_name": "आपका नाम",
        "location": "स्थान",
        "rating": "रेटिंग",
        "feedback": "प्रतिक्रिया",
        "submit": "जमा करें",
        "preferences": "प्राथमिकताएं",
        "theme": "थीम",
        "language": "भाषा",
        "data_management": "डेटा प्रबंधन",
        "clear_history": "स्कैन इतिहास साफ़ करें",
        "save_settings": "सेटिंग्स सुरक्षित करें",
        "settings_saved": "सेटिंग्स सफलतापूर्वक सहेजी गईं!",
        "footer_text": "गन्ने की पत्ती रोग डिटेक्टर | डीप लर्निंग द्वारा संचालित",
        "footer_sub": "स्मार्ट कृषि निर्णय लेने के लिए डिज़ाइन किया गया"
    },
    "Marathi": {
        "nav_dashboard": "डॅशबोर्ड",
        "nav_detection": "रोग शोधणे",
        "nav_analytics": "विश्लेषण",
        "nav_feedback": "अभिप्राय",
        "nav_settings": "सेटिंग्ज",
        "nav_header": "नेव्हिगेशन",
        "quick_stats": "द्रुत आकडेवारी",
        "total_scans": "एकूण स्कॅन",
        "healthy_leaves": "निरोगी पाने",
        "diseased_leaves": "रोगी पाने",
        "logout": "बाहेर पडा (लॉगआउट)",
        "login": "लॉगिन",
        "signup": "खाते तयार करा",
        "email": "ईमेल पत्ता",
        "password": "पासवर्ड",
        "name": "पूर्ण नाव",
        "confirm_password": "पासवर्डची पुष्टी करा",
        "register": "नोंदणी करा",
        "back_to_login": "लॉगिनवर परत जा",
        "sys_dashboard": "सिस्टम डॅशबोर्ड",
        "active_days": "सक्रिय दिवस",
        "recent_history": "अलीकडील स्कॅन इतिहास",
        "no_scans": "कोणतेही स्कॅन उपलब्ध नाहीत. आपला पहिला स्कॅन करण्यासाठी 'रोग शोधणे' वर जा.",
        "disease_detection": "रोग शोधणे",
        "upload_instruction": "पूर्वानुमानासाठी उसाच्या पानाचा फोटो अपलोड किंवा कॅप्चर करा",
        "choose_image": "पानाचा फोटो निवडा",
        "take_photo": "फोटो काढा",
        "uploaded_image": "अपलोड केलेला फोटो",
        "analysis_result": "विश्लेषण निकाल",
        "processing": "प्रक्रिया सुरू आहे...",
        "predicted_disease": "संभाव्य रोग",
        "confidence": "विश्वासार्हता",
        "model_confidence": "मॉडेल विश्वासार्हता",
        "high_conf": "उच्च मॉडेल विश्वासार्हता",
        "mod_conf": "मध्यम मॉडेल विश्वासार्हता",
        "low_conf": "कमी मॉडेल विश्वासार्हता - कृपया अधिक फोटोंसह सत्यापित करा.",
        "inference_time": "अनुमान वेळ",
        "disease_info": "रोगाची माहिती",
        "gradcam_heatmap": "Grad-CAM हीटमॅप (मॉडेल फोकस क्षेत्र)",
        "gradcam_caption": "पूर्वानुमानासाठी सर्वात प्रभावशाली क्षेत्र",
        "gradcam_error": "या मॉडेल कॉन्फिगरेशनसाठी Grad-CAM व्हिज्युअलायझेशन उपलब्ध नाही.",
        "analytics_title": "विश्लेषण डॅशबोर्ड",
        "disease_dist": "रोग वितरण (डोनट)",
        "conf_dist": "विश्वासार्हता वितरण",
        "monthly_trend": "मासिक स्कॅन ट्रेंड क्षेत्र आलेख",
        "community_fb": "वापरकर्ता अभिप्राय बोर्ड",
        "submit_fb": "तुमचा अभिप्राय सामायिक करा",
        "your_name": "तुमचे नाव",
        "location": "स्थान",
        "rating": "रेटिंग",
        "feedback": "अभिप्राय",
        "submit": "सादर करा",
        "preferences": "प्राधान्ये",
        "theme": "थीम",
        "language": "भाषा",
        "data_management": "डेटा व्यवस्थापन",
        "clear_history": "स्कॅन इतिहास साफ करा",
        "save_settings": "सेटिंग्ज जतन करा",
        "settings_saved": "सेटिंग्ज यशस्वीरित्या जतन केल्या!",
        "footer_text": "उसाच्या पानांच्या रोगांचे वर्गीकरण | डीप लर्निंगद्वारे समर्थित",
        "footer_sub": "स्मार्ट कृषी निर्णय घेण्यासाठी डिझाइन केलेले"
    },
    "Tamil": {
        "nav_dashboard": "டாஷ்போர்டு",
        "nav_detection": "நோய் கண்டறிதல்",
        "nav_analytics": "பகுப்பாய்வு",
        "nav_feedback": "கருத்து",
        "nav_settings": "அமைப்புகள்",
        "nav_header": "வழிசெலுத்தல்",
        "quick_stats": "விரைவான புள்ளிவிவரங்கள்",
        "total_scans": "மொத்த ஸ்கேன்கள்",
        "healthy_leaves": "ஆரோக்கியமான இலைகள்",
        "diseased_leaves": "நோய்வாய்ப்பட்ட இலைகள்",
        "logout": "வெளியேறு",
        "login": "உள்நுழை",
        "signup": "கணக்கை உருவாக்கு",
        "email": "மின்னஞ்சல் முகவரி",
        "password": "கடவுச்சொல்",
        "name": "முழு பெயர்",
        "confirm_password": "கடவுச்சொல்லை உறுதிப்படுத்து",
        "register": "பதிவு செய்",
        "back_to_login": "உள்நுழைவுக்குத் திரும்பு",
        "sys_dashboard": "கணினி டாஷ்போர்டு",
        "active_days": "செயலில் உள்ள நாட்கள்",
        "recent_history": "சமீபத்திய ஸ்கேன் வரலாறு",
        "no_scans": "ஸ்கேன்கள் எதுவும் இல்லை. உங்கள் முதல் ஸ்கேனைச் செய்ய 'நோய் கண்டறிதல்' பகுதிக்குச் செல்லவும்.",
        "disease_detection": "நோய் கண்டறிதல்",
        "upload_instruction": "கணிப்புக்கு கரும்பு இலையின் படத்தை பதிவேற்றவும் அல்லது படம் பிடிக்கவும்",
        "choose_image": "இலை படத்தை தேர்வு செய்யவும்",
        "take_photo": "படம் பிடிக்கவும்",
        "uploaded_image": "பதிவேற்றப்பட்ட படம்",
        "analysis_result": "பகுப்பாய்வு முடிவு",
        "processing": "செயலாக்கப்படுகிறது...",
        "predicted_disease": "கணிக்கப்பட்ட நோய்",
        "confidence": "நம்பகத்தன்மை",
        "model_confidence": "மாதிரி நம்பகத்தன்மை",
        "high_conf": "அதிக மாதிரி நம்பகத்தன்மை",
        "mod_conf": "மிதமான மாதிரி நம்பகத்தன்மை",
        "low_conf": "குறைந்த மாதிரி நம்பகத்தன்மை - கூடுதல் படங்களுடன் சரிபார்க்கவும்.",
        "inference_time": "கணிப்பு நேரம்",
        "disease_info": "நோய் தகவல்",
        "gradcam_heatmap": "Grad-CAM ஹீட்மேப் (மாதிரி கவனம் செலுத்தும் பகுதிகள்)",
        "gradcam_caption": "கணிப்புக்கு மிக முக்கியமான பகுதிகள்",
        "gradcam_error": "இந்த மாதிரி உள்ளமைவுக்கு Grad-CAM காட்சிப்படுத்தல் கிடைக்கவில்லை.",
        "analytics_title": "பகுப்பாய்வு டாஷ்போர்டு",
        "disease_dist": "நோய் பரவல் (வட்ட வரைபடம்)",
        "conf_dist": "நம்பகத்தன்மை பரவல்",
        "monthly_trend": "மாதாந்திர ஸ்கேன் போக்கு பகுதி வரைபடம்",
        "community_fb": "பயனர் கருத்து பலகை",
        "submit_fb": "உங்கள் கருத்தைப் பகிர்ந்து கொள்ளுங்கள்",
        "your_name": "உங்கள் பெயர்",
        "location": "இருப்பிடம்",
        "rating": "மதிப்பீடு",
        "feedback": "கருத்து",
        "submit": "சமர்ப்பி",
        "preferences": "விருப்பங்கள்",
        "theme": "தீம்",
        "language": "மொழி",
        "data_management": "தரவு மேலாண்மை",
        "clear_history": "ஸ்கேன் வரலாற்றை அழி",
        "save_settings": "அமைப்புகளைச் சேமி",
        "settings_saved": "அமைப்புகள் வெற்றிகரமாக சேமிக்கப்பட்டன!",
        "footer_text": "கரும்பு இலை நோய் கண்டறிதல் | ஆழ்ந்த கற்றல் மூலம் இயக்கப்படுகிறது",
        "footer_sub": "சிறந்த விவசாய முடிவுகளை எடுக்க வடிவமைக்கப்பட்டுள்ளது"
    },
    "Telugu": {
        "nav_dashboard": "డాష్‌బోర్డ్",
        "nav_detection": "తెగులు గుర్తింపు",
        "nav_analytics": "విశ్లేషణ",
        "nav_feedback": "అభిప్రాయం",
        "nav_settings": "సెట్టింగులు",
        "nav_header": "నావిగేషన్",
        "quick_stats": "త్వరిత గణాంకాలు",
        "total_scans": "మొత్తం స్కాన్‌లు",
        "healthy_leaves": "ఆరోగ్యకరమైన ఆకులు",
        "diseased_leaves": "తెగులు సోకిన ఆకులు",
        "logout": "లాగ్ అవుట్",
        "login": "లాగిన్",
        "signup": "ఖాతా సృష్టించండి",
        "email": "ఇమెయిల్ చిరునామా",
        "password": "పాసవర్డ్",
        "name": "పూర్తి పేరు",
        "confirm_password": "పాస్‌వర్డ్‌ను నిర్ధారించండి",
        "register": "నమోదు చేయండి",
        "back_to_login": "తిరిగి లాగిన్‌కు",
        "sys_dashboard": "సిస్టమ్ డాష్‌బోర్డ్",
        "active_days": "క్రియాశీల రోజులు",
        "recent_history": "ఇటీవలి స్కాన్ చరిత్ర",
        "no_scans": "స్కాన్‌లు అందుబాటులో లేవు. మీ మొదటి స్కాన్ చేయడానికి 'తెగులు గుర్తింపు' కి వెళ్ళండి.",
        "disease_detection": "తెగులు గుర్తింపు",
        "upload_instruction": "తెగులు గుర్తింపు కోసం చెరకు ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా ఫోటో తీయండి",
        "choose_image": "ఆకు చిత్రాన్ని ఎంచుకోండి",
        "take_photo": "ఫోటో తీయండి",
        "uploaded_image": "అప్‌లోడ్ చేసిన చిత్రం",
        "analysis_result": "విశ్లేషణ ఫలితం",
        "processing": "ప్రక్రియ జరుగుతోంది...",
        "predicted_disease": "గుర్తించిన తెగులు",
        "confidence": "సటిక్కత",
        "model_confidence": "మోడల్ సటిక్కత",
        "high_conf": "అధిక మోడల్ సటిక్కత",
        "mod_conf": "మధ్యస్థ మోడల్ సటిక్కత",
        "low_conf": "తక్కువ మోడల్ సటిక్కత - దయచేసి మరిన్ని చిత్రాలతో సరిచూసుకోండి.",
        "inference_time": "విశ్లేషణ సమయం",
        "disease_info": "తెగులు సమాచారం",
        "gradcam_heatmap": "Grad-CAM హీట్‌మ్యాప్ (మోడల్ దృష్టి పెట్టిన ప్రాంతాలు)",
        "gradcam_caption": "గుర్తింపుకు అత్యంత ప్రభావితమైన ప్రాంతాలు",
        "gradcam_error": "ఈ మోడల్ కాన్ఫిగరేషన్ కోసం Grad-CAM అందుబాటు లేదు.",
        "analytics_title": "విశ్లేషణ డాష్‌బోర్డ్",
        "disease_dist": "తెగులు విస్తరణ (పై చార్ట్)",
        "conf_dist": "సటిక్కత విస్తరణ",
        "monthly_trend": "నెలవారీ స్కాన్ ట్రెండ్ ఏరియా గ్రాఫ్",
        "community_fb": "వినియోగదారు అభిప్రాయాల బోర్డు",
        "submit_fb": "మీ అభిప్రాయాన్ని పంచుకోండి",
        "your_name": "మీ పేరు",
        "location": "ప్రాంతం",
        "rating": "రేటింగ్",
        "feedback": "అభిప్రాయం",
        "submit": "సమర్పించు",
        "preferences": "ప్రాధాన్యతలు",
        "theme": "థీమ్",
        "language": "భాష",
        "data_management": "డేటా నిర్వహణ",
        "clear_history": "స్కాన్ చరిత్రను తుడిచివేయి",
        "save_settings": "సెట్టింగులను భద్రపరచు",
        "settings_saved": "సెట్టింగులు విజయవంతంగా భద్రపరచబడ్డాయి!",
        "footer_text": "చెరకు ఆకు తెగులు గుర్తింపు | డీప్ లెర్నింగ్ ద్వారా నడుస్తుంది",
        "footer_sub": "తెలివైన వ్యవసాయ నిర్ణయాల కోసం రూపొందించబడింది"
    }
}

# TRANSLATE HELPER FUNCTION
def translate(key):
    lang = st.session_state.get("language", "English")
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

# DYNAMIC THEME & CSS INJECTOR
def inject_custom_css():
    theme = st.session_state.get("theme", "Light")
    if theme == "Dark":
        bg_color = "#121212"
        sidebar_bg = "#1E1E1E"
        sidebar_border = "#2D3748"
        text_color = "#E2E8F0"
        card_bg = "#1A1A1A"
        card_border = "#2D3748"
        card_shadow = "0 8px 32px rgba(0, 0, 0, 0.5)"
        input_bg = "#2D3748"
        input_text = "#F7FAFC"
        input_border = "#4A5568"
        radio_label_bg = "#1A1A1A"
        radio_label_border = "#2D3748"
        radio_label_hover_bg = "#2D3748"
        radio_label_hover_text = "#00E676"
        radio_label_hover_border = "#2E7D32"
        radio_label_active_bg = "linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%)"
        radio_label_active_border = "#00E676"
    else: # Light or System Default
        bg_color = "#ffffff"
        sidebar_bg = "#f9fbf9"
        sidebar_border = "#e8ede8"
        text_color = "#2D3748"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        card_shadow = "0 8px 32px rgba(31, 38, 135, 0.05)"
        input_bg = "#ffffff"
        input_text = "#2D3748"
        input_border = "#cbd5e0"
        radio_label_bg = "#f8f9fa"
        radio_label_border = "#e2e8f0"
        radio_label_hover_bg = "#f1f8e9"
        radio_label_hover_text = "#2E7D32"
        radio_label_hover_border = "#c8e6c9"
        radio_label_active_bg = "linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)"
        radio_label_active_border = "#1B5E20"

    css_code = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Hide Streamlit default UI components (branding, main menu, header bar, and developer/AI assistant icon) */
    #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        visibility: hidden !important;
        display: none !important;
        height: 0 !important;
        width: 0 !important;
    }}
    .stAppDeployButton, div.viewerBadge, button[title*="developer"], button[title*="developer tools"], button[aria-label*="developer"], iframe[title="widget"], button[title="Help"], button[aria-label="Help"], [data-testid="stConnectionStatus"], button[aria-label="show developer tools"], button[title="show developer tools"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
    }}

    /* Global Fonts and background */
    html, body, [class*="css"], .stApp {{
        font-family: 'Outfit', sans-serif !important;
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}


    /* Specific elements color override */
    h1, h2, h3, h4, h5, h6, p, label, span, div, li, small, .stMarkdown, [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}

    .main-title {{
        font-size: 3.2rem;
        background: linear-gradient(135deg, #1B5E20 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }}

    .sub-title {{
        text-align: center;
        font-size: 1.2rem;
        color: #4CAF50 !important;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }}

    /* Primary Buttons Styling */
    .stButton>button {{
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 25px !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.3) !important;
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%) !important;
    }}

    /* Sidebar Navigation Overrides */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}

    /* Hide the radio widget header label */
    [data-testid="stRadio"] > label {{
        display: none !important;
    }}

    /* Style the radio options container */
    [data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 10px !important;
        padding: 5px 0 !important;
    }}

    /* Style individual radio option labels */
    [data-testid="stRadio"] div[role="radiogroup"] label {{
        padding: 14px 20px !important;
        border-radius: 12px !important;
        background-color: {radio_label_bg} !important;
        border: 1px solid {radio_label_border} !important;
        color: {text_color} !important;
        font-weight: 500 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin: 0px !important;
        width: 100% !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    }}

    /* Hide the default radio circle/dot marker */
    [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {{
        display: none !important;
    }}

    /* Hover effect */
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {{
        background-color: {radio_label_hover_bg} !important;
        color: {radio_label_hover_text} !important;
        border-color: {radio_label_hover_border} !important;
        padding-left: 26px !important; /* Elegant slide effect */
        box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    }}

    /* Active selected state */
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
        background: {radio_label_active_bg} !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding-left: 26px !important;
        border-left: 6px solid {radio_label_active_border} !important;
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.25) !important;
        border-color: transparent !important;
    }}

    /* Ensure text inside active item is white */
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) * {{
        color: #ffffff !important;
    }}

    /* Sidebar logout button */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        background: transparent !important;
        color: #e53e3e !important;
        border: 1px solid #fed7d7 !important;
        border-radius: 12px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}

    section[data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #fff5f5 !important;
        border-color: #e53e3e !important;
        color: #c53030 !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* Content Cards styling */
    .card {{
        background-color: {card_bg} !important;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: {card_shadow};
        margin-bottom: 1.5rem;
        border: 1px solid {card_border} !important;
        border-left: 6px solid #2E7D32 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }}

    .stat-card {{
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.15);
    }}

    .stat-card * {{
        color: white !important;
    }}

    .prediction-card {{
        background-color: {radio_label_hover_bg} !important;
        border: 1px solid {radio_label_hover_border} !important;
        padding: 1.75rem;
        border-radius: 16px;
        margin: 1.5rem 0;
    }}

    .disease-tag {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 5px;
    }}

    .footer {{
        text-align: center;
        color: #2E7D32 !important;
        font-size: 0.95rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid {sidebar_border} !important;
    }}

    /* Streamlit Selectbox / Input elements custom color */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-color: {card_border} !important;
    }}
    
    /* Streamlit DataFrame style overrides */
    div[data-testid="stDataFrame"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# SESSION STATE INIT
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = []

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "language" not in st.session_state:
    st.session_state.language = "English"

# Apply theme-based CSS overrides
inject_custom_css()

# DATABASE CONFIGURATION & OPERATIONS (SQLite)
import sqlite3

DB_FILE = os.path.join(os.path.dirname(__file__), "sugarcane_detector.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            date TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            rating INTEGER NOT NULL,
            feedback TEXT NOT NULL,
            date TEXT NOT NULL,
            user_email TEXT
        )
    """)
    
    # Migration to add user_email column if it does not exist
    cursor.execute("PRAGMA table_info(feedback)")
    columns = [col[1] for col in cursor.fetchall()]
    if "user_email" not in columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN user_email TEXT")
    
    # Seed admin user & mock data if db is empty
    cursor.execute("SELECT * FROM users WHERE email = ?", ("admin@gmail.com",))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)", 
                       ("admin@gmail.com", "Admin", "1234"))
        
        # Mock scan history
        diseases = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
        for i in range(10):
            date_str = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO scans (user_email, date, disease, confidence) VALUES (?, ?, ?, ?)",
                           ("admin@gmail.com", date_str, random.choice(diseases), round(random.uniform(75, 98), 2)))
            
        # Mock feedback
        feedbacks = [
            ("Raj Kumar", "Maharashtra", 5, "Accurate detection helped in early intervention."),
            ("Priya Sharma", "Karnataka", 4, "Useful tool for crop monitoring."),
            ("Amit Patel", "Uttar Pradesh", 5, "Interface is clean and easy to use."),
            ("Sunil Reddy", "Tamil Nadu", 4, "The insights are highly beneficial."),
            ("Laxmi Nair", "Andhra Pradesh", 5, "Confidence score is helpful."),
            ("Karthik M", "Punjab", 4, "Great tool for farmers.")
        ]
        for name, loc, rating, text in feedbacks:
            date_str = (datetime.now() - timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("INSERT INTO feedback (name, location, rating, feedback, date) VALUES (?, ?, ?, ?, ?)",
                           (name, loc, rating, text, date_str))
            
    conn.commit()
    conn.close()

def add_user(email, name, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)", (email, name, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_user(email, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user

def add_scan(user_email, disease, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO scans (user_email, date, disease, confidence) VALUES (?, ?, ?, ?)",
                   (user_email, date_str, disease, confidence))
    conn.commit()
    conn.close()

def get_scans(user_email):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT date, disease, confidence FROM scans WHERE user_email = ? ORDER BY date DESC", (user_email,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for i, row in enumerate(rows):
        history.append({
            "date": datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"),
            "disease": row[1],
            "confidence": row[2],
            "image_id": f"img_{i+1}"
        })
    return history

def get_user_feedback(user_email):
    if not user_email:
        return None
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, location, rating, feedback, date FROM feedback WHERE user_email = ?", (user_email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "name": row[0],
            "location": row[1],
            "rating": row[2],
            "feedback": row[3],
            "date": datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")
        }
    return None

def save_or_update_feedback(user_email, name, location, rating, feedback_text):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if user_email:
        cursor.execute("SELECT id FROM feedback WHERE user_email = ?", (user_email,))
        row = cursor.fetchone()
        if row:
            cursor.execute("""
                UPDATE feedback 
                SET name = ?, location = ?, rating = ?, feedback = ?, date = ? 
                WHERE user_email = ?
            """, (name, location, rating, feedback_text, date_str, user_email))
            conn.commit()
            conn.close()
            return
            
    cursor.execute("""
        INSERT INTO feedback (name, location, rating, feedback, date, user_email) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, location, rating, feedback_text, date_str, user_email))
    conn.commit()
    conn.close()

def get_feedbacks():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, location, rating, feedback, date FROM feedback ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    
    feedbacks = []
    for row in rows:
        feedbacks.append({
            "name": row[0],
            "location": row[1],
            "rating": row[2],
            "feedback": row[3],
            "date": datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")
        })
    return feedbacks

# Initialize database
init_db()

# Pre-populate feedback and login email state
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if not st.session_state.user_feedback:
    st.session_state.user_feedback = get_feedbacks()

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.markdown(f"<h1 class='main-title'>{translate('login')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>{translate('upload_instruction')}</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input(translate('email'))
        password = st.text_input(translate('password'), type="password")
        login = st.form_submit_button(translate('login'))

        if login:
            user = check_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.page = "dashboard"
                st.session_state.prediction_history = get_scans(email)
                st.session_state.user_feedback = get_feedbacks()
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    if st.button(translate('signup')):
        st.session_state.page = "signup"
        st.rerun()

# ---------------- SIGNUP PAGE ----------------
def signup_page():
    st.markdown(f"<h1 class='main-title'>{translate('signup')}</h1>", unsafe_allow_html=True)

    with st.form("signup_form"):
        name = st.text_input(translate('name'))
        email = st.text_input(translate('email'))
        password = st.text_input(translate('password'), type="password")
        confirm = st.text_input(translate('confirm_password'), type="password")
        submit = st.form_submit_button(translate('register'))

        if submit:
            if not email or not password:
                st.error("Email and password are required.")
            elif password == confirm:
                if add_user(email, name, password):
                    st.success("Account created successfully.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Email already exists.")
            else:
                st.error("Passwords do not match.")

    if st.button(translate('back_to_login')):
        st.session_state.page = "login"
        st.rerun()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "resnet50_sugarcane_leaf_model.keras")
    if not os.path.exists(model_path):
        st.error(f" Model file not found at: {model_path}\n"
                 "Please check if the model file is in the project directory.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()
labels = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

# ---------------- GRAD-CAM FUNCTIONS ----------------
def make_gradcam_heatmap(img_arr, model, last_conv="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_arr)
        if isinstance(preds, list):
            preds = preds[0]
        class_id = tf.argmax(preds[0])
        loss = preds[:, class_id]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-10)

    return heatmap

def overlay_gradcam(original_pil_img, heatmap, alpha=0.4):

    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (original_pil_img.size[0], original_pil_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(original_pil_img)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(heatmap, alpha, img_bgr, 1 - alpha, 0)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed)

# ---------------- DASHBOARD ROOT ----------------
def dashboard():
    with st.sidebar:
        st.header(translate("nav_header"))
        
        # Display translated navigation menu
        selected = st.radio("Menu", [
            translate("nav_dashboard"),
            translate("nav_detection"),
            translate("nav_analytics"),
            translate("nav_feedback"),
            translate("nav_settings")
        ])

        st.markdown("---")
        st.subheader(translate("quick_stats"))
        total_scans = len(st.session_state.prediction_history)
        st.metric(translate("total_scans"), total_scans)

        if total_scans > 0:
            df_stats = pd.DataFrame(st.session_state.prediction_history)
            healthy_count = (df_stats["disease"] == "Healthy").sum()
            diseased_count = total_scans - healthy_count
        else:
            healthy_count = diseased_count = 0

        st.metric(translate("healthy_leaves"), healthy_count)
        st.metric(translate("diseased_leaves"), diseased_count)

        if st.button(translate("logout")):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    # Map translated options back to standard keys for routing
    option_map = {
        translate("nav_dashboard"): "Dashboard",
        translate("nav_detection"): "Disease Detection",
        translate("nav_analytics"): "Analytics",
        translate("nav_feedback"): "Feedback",
        translate("nav_settings"): "Settings"
    }
    page_name = option_map.get(selected, "Dashboard")

    if page_name == "Dashboard":
        show_dashboard()
    elif page_name == "Disease Detection":
        show_disease_detection()
    elif page_name == "Analytics":
        show_analytics()
    elif page_name == "Feedback":
        show_feedback()
    elif page_name == "Settings":
        show_settings()


# ---------------- DASHBOARD HOME ----------------
def show_dashboard():
    st.markdown(f"<h1 class='main-title'>{translate('sys_dashboard')}</h1>", unsafe_allow_html=True)

    history = st.session_state.prediction_history
    total_scans = len(history)

    if total_scans > 0:
        df = pd.DataFrame(history)
        healthy_count = (df["disease"] == "Healthy").sum()
        diseased_count = total_scans - healthy_count
        active_days = df["date"].dt.date.nunique()
    else:
        healthy_count = diseased_count = active_days = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"<div class='stat-card'><h3>{translate('total_scans')}</h3><h2>{total_scans}</h2></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div class='stat-card'><h3>{translate('healthy_leaves')}</h3><h2>{healthy_count}</h2></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div class='stat-card'><h3>{translate('diseased_leaves')}</h3><h2>{diseased_count}</h2></div>",
        unsafe_allow_html=True
    )
    col4.markdown(
        f"<div class='stat-card'><h3>{translate('active_days')}</h3><h2>{active_days}</h2></div>",
        unsafe_allow_html=True
    )

    st.subheader(translate('recent_history'))
    if total_scans:
        df_recent = pd.DataFrame(history[:5])
        st.dataframe(df_recent)
    else:
        st.info(translate('no_scans'))

# ---------------- DISEASE DETECTION PAGE ----------------
def show_disease_detection():
    st.markdown(f"<h1 class='main-title'>{translate('disease_detection')}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='sub-title'>{translate('upload_instruction')}</p>", unsafe_allow_html=True)

    tab_upload, tab_cam = st.tabs([translate("choose_image"), translate("take_photo")])
    file_bytes = None

    with tab_upload:
        uploaded_file = st.file_uploader(translate("choose_image"), type=["jpg", "jpeg", "png"], key="disease_file_uploader", label_visibility="collapsed")
        if uploaded_file:
            file_bytes = uploaded_file.read()

    with tab_cam:
        camera_file = st.camera_input(translate("take_photo"), key="disease_camera_input", label_visibility="collapsed")
        if camera_file:
            file_bytes = camera_file.read()

    if file_bytes:
        col1, col2 = st.columns(2)

        # Left: Display image
        with col1:
            st.subheader(translate('uploaded_image'))
            img_display = image.load_img(BytesIO(file_bytes), target_size=(224,224))
            st.image(img_display, use_container_width=True)

        # Right: Prediction
        with col2:
            st.subheader(translate('analysis_result'))
            with st.spinner(translate('processing')):
                # Real-time prediction speed measurement
                img_model = image.load_img(BytesIO(file_bytes), target_size=(224, 224))
                arr = image.img_to_array(img_model)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)

                start_time = time.time()
                pred = model.predict(arr)
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000

                idx = np.argmax(pred)
                conf = float(np.max(pred) * 100)
                disease = labels[idx]

            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.write(f"{translate('predicted_disease')}: **{disease}**")
            st.write(f"{translate('confidence')}: **{conf:.2f}%**")

            # Confidence Score Visualization
            st.write(translate('model_confidence'))
            st.progress(conf / 100.0)

            if conf >= 90:
                st.success(translate('high_conf'))
            elif conf >= 75:
                st.warning(translate('mod_conf'))
            else:
                st.error(translate('low_conf'))

            st.write(f"{translate('inference_time')}: **{inference_time_ms:.1f} ms**")
            st.markdown("</div>", unsafe_allow_html=True)

            # Disease Information
            st.subheader(translate('disease_info'))
            display_disease_info(disease)

            # Grad-CAM Heatmap
            st.subheader(translate('gradcam_heatmap'))
            try:
                heatmap = make_gradcam_heatmap(arr, model)
                gradcam_img = overlay_gradcam(img_display, heatmap)
                st.image(gradcam_img, use_container_width=True,
                         caption=translate('gradcam_caption'))
            except Exception as e:
                st.info(translate('gradcam_error'))
                st.text(f"Details: {str(e)}")

            # Update prediction history in database and sync session state
            add_scan(st.session_state.user_email, disease, conf)
            st.session_state.prediction_history = get_scans(st.session_state.user_email)


# ---------------- DISEASE INFORMATION ----------------
def display_disease_info(disease):
    info = {
        "Healthy": {
            "desc": "The leaf appears healthy with no signs of abnormality.",
            "symptoms": ["Uniform green color", "No visible spots"],
            "treatment": "No action required."
        },
        "Mosaic": {
            "desc": "Mosaic virus causes light and dark patches.",
            "symptoms": ["Mottled appearance", "Patchy colors"],
            "treatment": "Use disease-free planting material."
        },
        "RedRot": {
            "desc": "Red Rot is a harmful fungal disease.",
            "symptoms": ["Internal red discoloration", "Wilting"],
            "treatment": "Apply recommended fungicide and remove infected canes."
        },
        "Rust": {
            "desc": "Rust creates orange/brown pustules.",
            "symptoms": ["Brown powdery spots", "Premature leaf death"],
            "treatment": "Use resistant varieties and systemic fungicide."
        },
        "Yellow": {
            "desc": "Yellow leaf syndrome affects nutrient uptake.",
            "symptoms": ["Yellowing", "Reduced vigor"],
            "treatment": "Micronutrient supplementation and balanced fertilization required."
        }
    }

    d = info[disease]
    st.write("**Description:**", d["desc"])
    st.write("**Symptoms:**")
    for s in d["symptoms"]:
        st.write("-", s)
    st.write("**Treatment:**", d["treatment"])

# ---------------- ANALYTICS PAGE ----------------
def apply_plotly_theme(fig):
    theme = st.session_state.get("theme", "Light")
    if theme == "Dark":
        bg_color = "rgba(0,0,0,0)"
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#E2E8F0"
        grid_color = "#2D3748"
    else:
        bg_color = "rgba(0,0,0,0)"
        paper_bg = "rgba(0,0,0,0)"
        font_color = "#2D3748"
        grid_color = "#E2E8F0"

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=paper_bg,
        font=dict(family="Outfit, sans-serif", color=font_color),
        margin=dict(l=30, r=30, t=30, b=30),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color)),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(color=font_color))
    )
    return fig

def show_analytics():
    st.markdown(f"<h1 class='main-title'>{translate('analytics_title')}</h1>", unsafe_allow_html=True)

    if not st.session_state.prediction_history:
        st.info("No prediction data available. Perform some scans first.")
        return

    df = pd.DataFrame(st.session_state.prediction_history)
    theme = st.session_state.get("theme", "Light")

    col1, col2 = st.columns(2)

    # Disease distribution donut chart
    with col1:
        st.subheader(translate('disease_dist'))
        counts = df["disease"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.45,
            color_discrete_sequence=["#2E7D32", "#4CAF50", "#81C784", "#A5D6A7", "#C8E6C9"]
        )
        fig.update_traces(
            textinfo='percent+label',
            marker=dict(line=dict(color='#1E1E1E' if theme == "Dark" else '#ffffff', width=2))
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Confidence distribution histogram
    with col2:
        st.subheader(translate('conf_dist'))
        fig = px.histogram(
            df,
            x="confidence",
            nbins=12,
            color_discrete_sequence=["#4CAF50"]
        )
        fig.update_layout(
            bargap=0.1,
            xaxis_title="Confidence (%)",
            yaxis_title="Count"
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly scan trend area graph
    st.subheader(translate('monthly_trend'))
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    trend = df.groupby("month").size().sort_index()
    fig = px.area(
        x=trend.index,
        y=trend.values,
        labels={"x": "Month", "y": "Total Scans"},
        color_discrete_sequence=["#2E7D32"]
    )
    fig.update_traces(line=dict(width=3, shape="spline"))
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


# ---------------- FEEDBACK PAGE ----------------
def show_feedback():
    st.markdown(f"<h1 class='main-title'>{translate('community_fb')}</h1>", unsafe_allow_html=True)

    for fb in st.session_state.user_feedback:
        stars = "⭐" * fb['rating']
        st.markdown(
            f"""
            <div class='card' style='margin-bottom: 1.2rem; border-left: 6px solid #4CAF50;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <h4 style='margin: 0; font-weight: 700; color: inherit;'>{fb['name']} <span style='font-weight: 400; font-size: 0.95rem; color: #888;'>({fb['location']})</span></h4>
                    <span style='color: #FFD700; font-size: 1.1rem;'>{stars}</span>
                </div>
                <p style='margin: 0.5rem 0; font-size: 1.05rem; line-height: 1.5; color: inherit;'>"{fb['feedback']}"</p>
                <div style='text-align: right;'>
                    <small style='color: #888; font-size: 0.85rem;'>{fb['date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    user_email = st.session_state.get("user_email", "")
    existing_fb = get_user_feedback(user_email)
    
    if existing_fb:
        st.subheader(translate('submit_fb') + " (Edit/Update)")
        default_name = existing_fb["name"]
        default_location = existing_fb["location"]
        default_rating = existing_fb["rating"]
        default_text = existing_fb["feedback"]
        btn_text = "Update Feedback"
    else:
        st.subheader(translate('submit_fb'))
        default_name = ""
        default_location = ""
        default_rating = 5
        default_text = ""
        btn_text = translate('submit')
        
    with st.form("fb_form"):
        name = st.text_input(translate('your_name'), value=default_name)
        location = st.text_input(translate('location'), value=default_location)
        rating = st.slider(translate('rating'), 1, 5, value=default_rating)
        text = st.text_area(translate('feedback'), value=default_text)

        submit = st.form_submit_button(btn_text)
        if submit and name and text:
            save_or_update_feedback(user_email, name, location, rating, text)
            st.session_state.user_feedback = get_feedbacks()
            st.success("Feedback updated successfully!" if existing_fb else "Feedback submitted successfully!")
            st.rerun()


# ---------------- SETTINGS PAGE ----------------
def show_settings():
    st.markdown(f"<h1 class='main-title'>{translate('settings')}</h1>", unsafe_allow_html=True)

    st.subheader(translate('preferences'))
    
    theme_list = ["Light", "Dark", "System Default"]
    lang_list = ["English", "Hindi", "Marathi", "Tamil", "Telugu"]
    
    current_theme = st.session_state.get("theme", "Light")
    current_lang = st.session_state.get("language", "English")
    
    theme_index = theme_list.index(current_theme) if current_theme in theme_list else 0
    lang_index = lang_list.index(current_lang) if current_lang in lang_list else 0
    
    theme_choice = st.selectbox(translate('theme'), theme_list, index=theme_index)
    lang_choice = st.selectbox(translate('language'), lang_list, index=lang_index)

    st.subheader(translate('data_management'))
    if st.button(translate('clear_history')):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scans WHERE user_email = ?", (st.session_state.user_email,))
        conn.commit()
        conn.close()
        st.session_state.prediction_history = []
        st.success("Scan history cleared.")
        st.rerun()

    if st.button(translate('save_settings')):
        st.session_state.theme = theme_choice
        st.session_state.language = lang_choice
        st.success(translate('settings_saved'))
        st.rerun()

# ---------------- ROUTING LOGIC ----------------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
else:
    dashboard()

# ---------------- FOOTER ----------------
st.markdown(f"""
<div class='footer'>
    <p>{translate('footer_text')}</p>
    <p>{translate('footer_sub')}</p>
</div>
""", unsafe_allow_html=True)
