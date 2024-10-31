import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# Load model dan preprocessing tools
label_encoder = joblib.load('label_encoder.joblib')
model = tf.keras.models.load_model('tourism_classifier.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# Load data tourism untuk filtering
tourism_df = pd.read_csv('tourism_with_id.csv')

# Filter dan pencarian berdasarkan nama tempat wisata
def filter_tourism(city, price_range):
    filtered_df = tourism_df[(tourism_df['City'] == city) &
                             (tourism_df['Price'] >= price_range[0]) &
                             (tourism_df['Price'] <= price_range[1])]
    return filtered_df

# Fungsi rekomendasi berdasarkan model
def recommend_tourism(selected_category):
    category_encoded = label_encoder.transform([selected_category])
    category_onehot = tf.keras.utils.to_categorical(category_encoded, num_classes=len(label_encoder.classes_))
    recommended_df = tourism_df[tourism_df['Category'] == selected_category]
    return recommended_df[['Place_Name', 'City', 'Price', 'Description']]

def predict_category(model, tfidf_vectorizer, scaler, label_encoder, place_name):
    # Membuat feature yang sama seperti data training
    name_word_count = len(place_name.split())
    
    combined_features = f"{place_name}  {name_word_count}"
    
    # Transformasi input menggunakan TF-IDF
    input_vector = tfidf_vectorizer.transform([combined_features]).toarray()
    
    # Normalisasi input menggunakan scaler yang sama dengan training
    input_scaled = scaler.transform(input_vector)
    
    # Lakukan prediksi
    prediction = model.predict(input_scaled)
    
    # Dapatkan probabilitas untuk setiap kategori
    probabilities = prediction[0]
    predicted_category_index = np.argmax(probabilities)
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]
    
    # Tampilkan probabilitas untuk setiap kategori
    categories = label_encoder.classes_
    for cat, prob in zip(categories, probabilities):
        print(f"{cat}: {prob:.4f}")
    
    return predicted_category

def get_recommendations(place_name, tourism_df, model, tfidf_vectorizer, scaler, label_encoder, n_recommendations=5):
    # Cari tempat wisata di dataset
    input_place = tourism_df[tourism_df['Place_Name'].str.lower() == place_name.lower()]
    
    if input_place.empty:
        # Jika tempat tidak ditemukan di dataset, prediksi kategorinya
        predicted_category = predict_category(
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            scaler=scaler,
            label_encoder=label_encoder,
            place_name=place_name
        )
        category = predicted_category
    else:
        # Jika tempat ditemukan, gunakan kategori yang ada
        category = input_place['Category'].iloc[0]
    
    # Filter tempat wisata dengan kategori yang sama
    similar_places = tourism_df[tourism_df['Category'] == category].copy()
    
    # Hapus tempat wisata input dari rekomendasi jika ada
    if not input_place.empty:
        similar_places = similar_places[similar_places['Place_Name'] != place_name]
    
    # Acak urutan dan ambil n rekomendasi
    recommendations = similar_places.sample(n=min(n_recommendations, len(similar_places)))
    
    # Format output
    output_recommendations = recommendations[[
        'Place_Name', 'Category', 'City', 'Price', 'Description',
    ]].copy()
    
    return output_recommendations

# UI Aplikasi Streamlit
st.title("Tourism Recommendation System")

# 1. Filter City dan Price Range
st.sidebar.header("Filter Options")
city_option = st.sidebar.selectbox("Select City", sorted(tourism_df['City'].unique()))
price_range = st.sidebar.slider("Select Price Range", 
                                min_value=int(tourism_df['Price'].min()), 
                                max_value=int(tourism_df['Price'].max()), 
                                value=(int(tourism_df['Price'].min()), int(tourism_df['Price'].max())))

filtered_places = filter_tourism(city_option, price_range)
st.write(f"*Filtered Places in {city_option}*")
st.write(filtered_places[['Place_Name', 'Price']])


# 3. Sistem Rekomendasi
st.write("### Recommendation System")
input_place = st.text_input("Enter a place you like:")

if st.button("Get Recommendations"):
    recommendations = get_recommendations(
        place_name=input_place,
        tourism_df=tourism_df,
        model=model,
        tfidf_vectorizer=tfidf_vectorizer,
        scaler=scaler,
        label_encoder=label_encoder,
        n_recommendations=5
    )
    
    if not recommendations.empty:
        st.write(f"*Recommended Places similar to {input_place}:*")
        st.dataframe(recommendations)
    else:
        st.write("No recommendations available.")