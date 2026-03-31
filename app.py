import streamlit as st
import os
import pandas as pd
import librosa
import numpy as np

from feature_extraction import extract_features
from rule_classifier import classify_gender, evaluate_predictions

st.set_page_config(page_title="Ses Sınıflandırma", page_icon="🎙️", layout="wide")

st.title("🎙️ Ses Analizi ve Cinsiyet Sınıflandırma Sistemi")
st.markdown("Zaman düzlemi analiz yöntemleri (STE, ZCR, Otokorelasyon) ve kural tabanlı algoritma kullanılarak Ses - Cinsiyet Tespiti.")

# Sidebar navigation
page = st.sidebar.selectbox("Gezinme", ["Bireysel Ses Analizi", "Veri Seti Başarı Analizi"])

if page == "Bireysel Ses Analizi":
    st.header("Tekil Ses Sınıflandırma")
    st.markdown("Lütfen analiz etmek istediğiniz bir `.wav` dosyasını yükleyin.")
    
    uploaded_file = st.file_uploader("Dosya Seç (WAV)", type=['wav'])
    
    if uploaded_file is not None:
        st.subheader("1. Orijinal Ses")
        st.audio(uploaded_file, format='audio/wav')
        
        # Save temp file
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Sinyal analiz ediliyor..."):
            audio, sr = librosa.load(temp_path, sr=22050)
            
            avg_f0, avg_zcr, avg_energy = extract_features(audio, sr, window_duration=0.03)
            prediction = classify_gender(avg_f0)
            
        st.subheader("2. Çıkarılan Öznitelikler")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ortalama F0 (Pitch)", f"{avg_f0:.1f} Hz")
        col2.metric("Ortalama ZCR", f"{avg_zcr:.4f}")
        col3.metric("Ortalama Enerji (VAD)", f"{avg_energy:.2f}")
        
        st.subheader("3. Sınıflandırma Sonucu")
        if prediction == "Erkek":
            st.info(f"Sonuç: **{prediction}** 👨")
        elif prediction == "Kadın":
            st.success(f"Sonuç: **{prediction}** 👩")
        else:
            st.warning(f"Sonuç: **{prediction}** 🧒")
            
        os.remove(temp_path)

elif page == "Veri Seti Başarı Analizi":
    st.header("Tüm Veri Seti Başarım Testi ve İstatistiksel Rapor")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.join(base_dir, "Dataset", "extracted_features.csv")
    
    if os.path.exists(feature_path):
        df = pd.read_csv(feature_path)
        st.markdown(f"**Toplam İşlenen Dosya Sayısı:** {len(df)}")
        
        acc, stats_df, cm, results_df = evaluate_predictions(df)
        
        st.subheader("1. Genel Sistem Başarısı")
        st.info(f"Accuracy Oranı: **%{acc:.2f}**")
        
        st.subheader("2. Sınıf Bazlı İstatistikler (Ortalama F0 ve Sapma)")
        st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("3. Karışıklık Matrisi (Confusion Matrix)")
        st.dataframe(cm)
        
        st.subheader("4. Öznitelik Tablosu (Örnek Veriler)")
        st.dataframe(results_df.head(15), use_container_width=True)
    else:
        st.warning("Veri seti henüz işlenmemiş! Lütfen terminalde `python feature_extraction.py` çalıştırarak özellikleri çıkarın.")
