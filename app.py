import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Set up the Streamlit app
st.set_page_config(page_title="Aplikasi Klasifikasi Random Forest", layout="wide")
st.title("Aplikasi Klasifikasi Random Forest")

# Load the dataset
@st.cache_data
def load_data():
    # Load the dataset from the provided CSV file
    data = pd.read_csv("Classification.csv")
    return data

data = load_data()

# Initialize variables
model = None
X_final = None
y = None
encoder = None
categorical_cols = None
X = None  # Initialize X to None

# Display the dataset
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih tampilan:", ['Beranda', 'Dataset', 'Pelatihan Model'])

if option == 'Beranda':
    st.header("Selamat datang di Aplikasi Klasifikasi Random Forest")
    st.write("Aplikasi ini memungkinkan Anda untuk melatih dan memvisualisasikan model klasifikasi menggunakan Random Forest.")
    st.write("Silakan pilih opsi di sidebar untuk melanjutkan.")
    
    # Additional text to emphasize the effectiveness of the algorithm
    st.subheader("Mengapa Random Forest?")
    st.write("""
        Algoritma Random Forest adalah salah satu metode klasifikasi yang paling kuat dan efektif. 
        Dengan menggunakan ensemble dari pohon keputusan, Random Forest dapat menangani data yang kompleks 
        dan memberikan hasil yang akurat. 
    """)
    st.write("""
        Aplikasi ini dirancang untuk membantu Anda memahami bagaimana Random Forest bekerja dengan dataset 
        yang Anda miliki. Dengan antarmuka yang sederhana, Anda dapat melatih model, melakukan prediksi, 
        dan melihat hasilnya dengan mudah. 
    """)
    st.write("""
        Kami percaya bahwa algoritma ini sangat cocok untuk dataset ini, karena kemampuannya dalam 
        menangani variabel yang beragam dan memberikan interpretasi yang jelas terhadap hasil klasifikasi.
    """)

elif option == 'Dataset':
    st.header("Tinjauan Dataset")
    st.write("Berikut adalah data:")
    st.write(data.head())
    st.write("Ringkasan Data:")
    st.write(data.describe())

elif option == 'Pelatihan Model':
    st.header("Pelatihan Model")
    
    # Prepare the data
    X = data.drop(columns=['Drug'])  # Features
    y = data['Drug']  # Target variable

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # OneHotEncoding for categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_encoded = encoder.fit_transform(X[categorical_cols])
    
    # Create a DataFrame for the encoded features
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine encoded features with the rest of the data
    X_final = pd.concat([X_encoded_df, X.drop(columns=categorical_cols).reset_index(drop=True)], axis=1)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

    # Model Training
    st.subheader("Pengaturan Model")
    n_estimators = st.slider("Jumlah Pohon", min_value=10, max_value=200, value=100, step=10)
    max_depth = st.slider("Kedalaman Maksimum", min_value=1, max_value=20, value=5, step=1)

    # Fit the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    st.write("Model Berhasil Dilatih!")
    y_pred = model.predict(X_test)
    st.write("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion Matrix.:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # Prediction Section
    st.subheader("Buat Prediksi")
    
    # Input fields for the features
    age = st.number_input("Usia", min_value =0, max_value=100, value=30)
    sex = st.selectbox("Jenis Kelamin", options=['F', 'M'])
    bp = st.selectbox("Tekanan Darah", options=['NORMAL', 'LOW', 'HIGH'])
    cholesterol = st.selectbox("Kadar Kolesterol", options=['NORMAL', 'HIGH'])
    na_to_k = st.number_input("Rasio Na/K", min_value=0.0, max_value=100.0, value=10.0)

    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'BP': [bp],
        'Cholesterol': [cholesterol],
        'Na_to_K': [na_to_k]
    })

    # Make prediction if the model is trained
    if model is not None and encoder is not None:
        # OneHotEncoding for the input features
        input_encoded = encoder.transform(input_data[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        input_final = pd.concat([input_encoded_df, input_data.drop(columns=categorical_cols).reset_index(drop=True)], axis=1)

        prediction = model.predict(input_final)
        st.write("Kelas yang Diprediksi:", prediction[0])
    else:
        st.write("Silakan latih model terlebih dahulu untuk membuat prediksi.")

# Enhancements for User Interface
st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.write("Aplikasi ini menggunakan algoritma Random Forest untuk klasifikasi data. Anda dapat melatih model dan melihat hasilnya dengan mudah.")

# Remove the footer from the sidebar
# Add footer at the bottom of the app
st.markdown("---")
st.markdown("### Terima kasih telah menggunakan aplikasi kami!")