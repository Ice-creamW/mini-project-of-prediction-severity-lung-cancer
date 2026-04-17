import streamlit as st
import pandas as pd
import pickle

# --- ส่วนหัวของเว็บ ---
st.write("""
# 🫁 ระบบทำนายความรุนแรงของมะเร็งปอด
กรุณากรอกข้อมูลส่วนบุคคลและอาการเพื่อวิเคราะห์เบื้องต้น
""")

# --- ส่วนรับข้อมูลจากผู้ใช้ (Sidebar หรือหน้าหลัก) ---
st.sidebar.header('ระบุข้อมูลของคุณ')

def user_input_features():
    age = st.sidebar.slider('อายุ', 18, 90, 30)
    smoking = st.sidebar.selectbox('ระดับการสูบบุหรี่ (0-8)', range(9))
    shortness_of_breath = st.sidebar.selectbox('ระดับความเหนื่อยหอบ (0-8)', range(9))
    # เพิ่มปัจจัยอื่นๆ ตาม Dataset ของคุณ
    
    data = {
        'AGE': age,
        'SMOKING': smoking,
        'SHORTNESS_OF_BREATH': shortness_of_breath
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- ส่วนการทำนาย ---
# โหลด Model ที่เราทำไว้
load_model = pickle.load(open('lung_cancer_model.pkl', 'rb'))

if st.button('วิเคราะห์ผล'):
    prediction = load_model.predict(input_df)
    st.subheader('ผลการวิเคราะห์:')
    
    # แสดงผลตามความรุนแรง
    if prediction[0] == 'High':
        st.error(f'ระดับความรุนแรง: {prediction[0]} (ควรพบแพทย์โดยด่วน)')
    elif prediction[0] == 'Medium':
        st.warning(f'ระดับความรุนแรง: {prediction[0]}')
    else:
        st.success(f'ระดับความรุนแรง: {prediction[0]}')