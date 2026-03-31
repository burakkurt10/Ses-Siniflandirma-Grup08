import pandas as pd
import numpy as np

def classify_gender(f0):
    """
    Kural Tabanlı (Rule-Based) Cinsiyet Sınıflandırıcı.
    F0 ortalama değerine (Hz) bakarak karar verir.
    """
    # Gerçek veri setinin hesaplanan ortalamalarına göre kurallar:
    # Erkek Ortalaması: ~173 Hz
    # Kadın Ortalaması: ~258 Hz
    # Çocuk Ortalaması: ~326 Hz
    if f0 <= 50:
        return "Bilinmiyor"
    elif f0 < 210:
        return "Erkek"
    elif f0 >= 210 and f0 < 285:
        return "Kadın"
    else:
        return "Çocuk"

def evaluate_predictions(features_df):
    """
    Veri seti üzerindeki başarı ve istatistikleri hesaplar.
    """
    if "Gercek_Cinsiyet" in features_df.columns:
        features_df["Gercek_Cinsiyet"] = features_df["Gercek_Cinsiyet"].apply(
            lambda g: "Erkek" if str(g).strip().upper() in ['M', 'E', 'ERKEK', 'MALE']
            else "Kadın" if str(g).strip().upper() in ['F', 'K', 'KADIN', 'FEMALE']
            else "Çocuk" if str(g).strip().upper() in ['C', 'Ç', 'COCUK', 'ÇOCUK', 'CHILD']
            else "Bilinmiyor"
        )
        
    features_df["Tahmin"] = features_df["Ortalama_F0"].apply(classify_gender)
    
    correct = sum(features_df["Tahmin"] == features_df["Gercek_Cinsiyet"])
    total = len(features_df)
    accuracy = correct / total if total > 0 else 0
    
    stats = []
    classes = ["Erkek", "Kadın", "Çocuk"]
    for cls in classes:
        subset = features_df[features_df["Gercek_Cinsiyet"] == cls]
        if len(subset) > 0:
            stats.append({
                "Sınıf": cls,
                "Örnek Sayısı": len(subset),
                "Ortalama F0 (Hz)": subset["Ortalama_F0"].mean(),
                "Standart Sapma": subset["Ortalama_F0"].std(ddof=0) if len(subset) > 1 else 0.0,
                "Başarı (%)": (sum(subset["Tahmin"] == cls) / len(subset)) * 100
            })
            
    stats_df = pd.DataFrame(stats)
    
    # Karışıklık Matrisi
    cm = pd.crosstab(
        features_df["Gercek_Cinsiyet"], 
        features_df["Tahmin"], 
        rownames=['Gerçek'], 
        colnames=['Tahmin']
    )
    
    return accuracy * 100, stats_df, cm, features_df

if __name__ == "__main__":
    import os
    feature_path = os.path.join(os.path.dirname(__file__), "Dataset", "extracted_features.csv")
    if os.path.exists(feature_path):
        df = pd.read_csv(feature_path)
        acc, stats_df, cm_df, result_df = evaluate_predictions(df)
        print(f"✅ Genel Başarı: %{acc:.2f}\n")
        print(stats_df.to_string(index=False))
        print("\nKarışıklık Matrisi:\n", cm_df)
