import os
import glob
import librosa
import numpy as np
import pandas as pd

def extract_features(audio, sr, window_duration=0.03):
    """
    Computes Short-Time Energy (STE), Zero Crossing Rate (ZCR), and F0 per frame.
    """
    frame_length = int(sr * window_duration)
    hop_length = frame_length // 2

    # Calculate Energy for each frame
    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio), hop_length)
    ])

    if len(energy) == 0:
        return 0.0, 0.0, 0.0

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Voiced thresholding based on energy
    threshold = 0.1 * np.max(energy) if np.max(energy) > 0 else 0
    voiced_indices = np.where(energy > threshold)[0]
    
    if len(voiced_indices) == 0:
        return 0.0, 0.0, 0.0

    avg_zcr_voiced = np.mean(zcr[voiced_indices])
    avg_energy_voiced = np.mean(energy[voiced_indices])

    # Pitch extraction per voiced frame
    valid_f0s = []
    
    for idx in voiced_indices:
        start = idx * hop_length
        end = start + frame_length
        frame = audio[start:end]
        
        if len(frame) < frame_length:
            continue
            
        # Hanning window reduces edge discontinuities
        frame = frame * np.hanning(len(frame))
        
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        min_lag = sr // 500  # 500Hz Max Pitch
        max_lag = sr // 50   # 50Hz Min Pitch
        
        if max_lag > len(autocorr): max_lag = len(autocorr)
        if min_lag >= len(autocorr): continue
            
        lag_range = autocorr[min_lag:max_lag]
        if len(lag_range) == 0: continue
            
        peak_idx = np.argmax(lag_range)
        peak_lag = peak_idx + min_lag
        
        # Valid pitch detection (peak intensity threshold)
        if lag_range[peak_idx] > 0.25 * autocorr[0]:
            f0 = sr / peak_lag
            valid_f0s.append(f0)
            
    avg_f0 = np.mean(valid_f0s) if valid_f0s else 0.0
    return avg_f0, avg_zcr_voiced, avg_energy_voiced

def process_dataset(dataset_dir="Dataset"):
    """
    Reads all metadata excel files, processes the .wav files, 
    and saves features to a dataframe.
    """
    excel_files = glob.glob(os.path.join(dataset_dir, "**", "*.xlsx"), recursive=True)
    if not excel_files:
        print("Excel meta verisi bulunamadı!")
        return None
        
    df_list = []
    for f in excel_files:
        temp_df = pd.read_excel(f)
        cols = [str(c).lower() for c in temp_df.columns]
        if 'cinsiyet' in cols or 'gender' in cols or 'dosya_adi' in cols or 'file name' in cols:
            df_list.append(temp_df)
            continue
            
        header_idx = None
        for idx, r in temp_df.head(5).iterrows():
            row_vals = [str(v).lower() for v in r.values]
            if any('cinsiyet' in str(v) or 'gender' in str(v) or 'dosya' in str(v) for v in row_vals):
                header_idx = idx
                break
                
        if header_idx is not None:
            temp_df.columns = temp_df.iloc[header_idx]
            temp_df = temp_df.iloc[header_idx+1:].reset_index(drop=True)
            
        df_list.append(temp_df)
        
    if not df_list:
        return None
        
    master_df = pd.concat(df_list, ignore_index=True)
    
    features = []
    
    for index, row in master_df.iterrows():
        filepath = None
        for col in master_df.columns:
            if "dosya" in str(col).lower() or "file" in str(col).lower():
                filepath = row.get(col)
                break
                
        cinsiyet = None
        for col in master_df.columns:
            if "cinsiyet" in str(col).lower() or "gender" in str(col).lower() or "cins" in str(col).lower():
                cinsiyet = row.get(col)
                break
        
        if pd.isna(filepath) or not filepath:
            continue
            
        filepath = str(filepath).strip()
        
        # Resolve absolute path issue if paths were relative
        if not os.path.exists(filepath):
            file_name = os.path.basename(filepath)
            search_result = glob.glob(os.path.join(dataset_dir, "**", file_name), recursive=True)
            if search_result:
                filepath = search_result[0]
            
        if not os.path.exists(filepath):
            print(f"HATA: Dosya bulunamadı - {file_name}")
            continue
            
        print(f"İşleniyor: {os.path.basename(filepath)}...")
        try:
            import soundfile as sf
            # Explicitly load with sf to avoid audioread hanging on corrupted files
            audio, sf_sr = sf.read(filepath)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1) # Convert to mono
                
            if sf_sr != 22050:
                audio = librosa.resample(y=audio, orig_sr=sf_sr, target_sr=22050)
            
            sr = 22050
            avg_f0, avg_zcr, avg_energy = extract_features(audio, sr, window_duration=0.03)
        except Exception as e:
            print(f"HATA: {os.path.basename(filepath)} bozuk veya okunamiyor. Atlanıyor. Hata: {str(e)}")
            continue
        
        features.append({
            "Dosya_Adı": os.path.basename(filepath),
            "Gercek_Cinsiyet": cinsiyet,
            "Ortalama_F0": avg_f0,
            "Ortalama_ZCR": avg_zcr,
            "Ortalama_Enerji": avg_energy
        })
        
    if features:
        features_df = pd.DataFrame(features)
        output_path = os.path.join(dataset_dir, "extracted_features.csv")
        features_df.to_csv(output_path, index=False)
        print(f"✅ Özellikler çıkarıldı ve kaydedildi: {output_path}")
        return features_df
    return None

if __name__ == "__main__":
    # Test script path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "Dataset")
    process_dataset(dataset_dir)
