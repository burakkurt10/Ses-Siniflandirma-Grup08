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
page = st.sidebar.selectbox("Gezinme", ["Bireysel Ses Analizi", "Veri Seti Başarı Analizi", "Algoritma Kıyaslaması (Otokorelasyon vs FFT)"])

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
            import soundfile as sf
            try:
                audio, sr = sf.read(temp_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != 22050:
                    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=22050)
                sr = 22050
            except:
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
        st.markdown("Veri setinden elde edilen istatistiksel **F0 değerlerinin dağılım tablosu** (E/K/Ç).")
        st.dataframe(stats_df, use_container_width=True)
        
        st.subheader("3. Karışıklık Matrisi (Confusion Matrix)")
        st.dataframe(cm)
        
        st.subheader("4. Öznitelik Tablosu (Örnek Veriler)")
        st.dataframe(results_df.head(15), use_container_width=True)
    else:
        st.warning("Veri seti henüz işlenmemiş! Lütfen terminalde `python feature_extraction.py` çalıştırarak özellikleri çıkarın.")

elif page == "Algoritma Kıyaslaması (Otokorelasyon vs FFT)":
    st.header("Otokorelasyon ve FFT (Hızlı Fourier Dönüşümü) Kıyaslaması")
    st.markdown("Zaman düzleminde çıkarılan **Otokorelasyon sonlanımları** ile frekans düzlemindeki **FFT (Fast Fourier Transform) frekans spektrumlarının** sinyal üzerindeki karşılıklı analizi ve kıyaslaması.")
    import matplotlib.pyplot as plt
    
    uploaded_file = st.file_uploader("Örnek Bir Ses Dosyası Seç (WAV)", type=['wav'])
    
    if uploaded_file is not None:
        temp_path = "temp_fft_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Grafikler oluşturuluyor..."):
            import soundfile as sf
            audio, sr = sf.read(temp_path)
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            
            # Seçili küçük bir ses çerçevesi alalım (örnek 0.05 saniye = 50ms)
            start = len(audio) // 2
            frame_length = int(sr * 0.05)
            frame = audio[start:start+frame_length]
            frame = frame * np.hanning(len(frame))
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
            
            # 1. Zaman Düzlemi
            axs[0].plot(np.linspace(0, 0.05, len(frame)), frame, color='blue')
            axs[0].set_title("1. Sinyalin Zaman Düzlemi (Time Domain)")
            axs[0].set_xlabel("Zaman (s)")
            axs[0].set_ylabel("Genlik")
            axs[0].grid(True)
            
            # 2. Otokorelasyon (Zaman Düzlemi Özellik Çıkarımı)
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            axs[1].plot(autocorr, color='green')
            axs[1].set_title("2. Otokorelasyon Fonksiyonu Sonucu")
            axs[1].set_xlabel("Gecikme (Lag)")
            axs[1].set_ylabel("Benzerlik (Korelasyon)")
            axs[1].grid(True)
            
            # 3. FFT (Frekans Düzlemi Özellik Çıkarımı)
            fft_result = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1/sr)
            axs[2].plot(freqs, fft_result, color='red')
            axs[2].set_title("3. FFT (Fast Fourier Transform) Frekans Spektrumu")
            axs[2].set_xlabel("Frekans (Hz)")
            axs[2].set_ylabel("Büyüklük (Magnitude)")
            axs[2].set_xlim(0, 1000) # İnsan sesi frekansları 1000Hz altındadır
            axs[2].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        os.remove(temp_path)
