import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. เชื่อมต่อ Google Sheet ---
# นำลิงก์ที่ก๊อปมาจาก Google Sheet (ที่ลงท้ายด้วย output=csv) มาวางในเครื่องหมายคำพูดด้านล่างนี้
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv" 

@st.cache_data(ttl=1) # ตั้งค่าให้ดึงข้อมูลใหม่ทันทีที่รีเฟรช
def load_data():
    try:
        df_sheet = pd.read_csv(SHEET_URL)
        # ตั้งชื่อคอลัมน์ให้ตรงกับที่ AI ต้องการ (เรียงตามลำดับใน Sheet)
        df_sheet.columns = ['Timestamp', 'Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        return df_sheet
    except Exception as e:
        st.error(f"⚠️ ไม่สามารถดึงข้อมูลได้: {e}")
        return pd.DataFrame()

df = load_data()
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- 2. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="SportMatch AI", layout="wide")
st.title("🏆 SportMatch AI: ค้นหากีฬาจากฐานข้อมูลจริง")
st.markdown("---")

# เลือกโหมด (2 Cases ตามที่คุณต้องการ)
mode = st.sidebar.selectbox("🎯 คุณคือใคร?", ["มือใหม่ (Newbie)", "นักกีฬา (อยากลองกีฬาใหม่)"])

if not df.empty:
    if mode == "มือใหม่ (Newbie)":
        st.header("🌟 แนะนำกีฬาเริ่มต้นสำหรับคุณ")
        col1, col2 = st.columns(2)
        with col1:
            u_intensity = st.slider("ระดับความเหนื่อยที่รับได้", 1, 10, 5)
            u_social = st.slider("ความชอบเข้าสังคม", 1, 10, 5)
        with col2:
            u_budget = st.slider("งบประมาณอุปกรณ์", 1, 10, 5)
            u_flex = st.slider("ความยืดหยุ่นร่างกาย", 1, 10, 5)

        if st.button("✨ วิเคราะห์ผล"):
            user_vector = np.array([[u_intensity, u_social, u_budget, u_flex, 5]])
            sim = cosine_similarity(user_vector, df[features])
            df['Score'] = sim[0]
            res = df.sort_values(by='Score', ascending=False).head(3)
            
            st.success("กีฬาที่แนะนำ:")
            for i, row in res.iterrows():
                st.write(f"### 🥇 {row['Sport']} (ความเหมาะสม {round(row['Score']*100, 1)}%)")

    else:
        st.header("🔄 ต่อยอดทักษะจากกีฬาเดิม")
        current_sport = st.selectbox("ตอนนี้เล่นกีฬาอะไรอยู่?", df['Sport'].unique())
        
        if st.button("🚀 หากีฬาใหม่ที่ใช้ทักษะคล้ายกัน"):
            target = df[df['Sport'] == current_sport][features].values
            others = df[df['Sport'] != current_sport].copy()
            sim = cosine_similarity(target, others[features])
            others['Score'] = sim[0]
            res = others.sort_values(by='Score', ascending=False).head(3)
            
            st.info(f"คุณน่าจะชอบกีฬาเหล่านี้เช่นกัน:")
            for i, row in res.iterrows():
                st.write(f"### 🔥 {row['Sport']} (ความคล้ายคลึง {round(row['Score']*100, 1)}%)")

    # ส่วน Admin ดูข้อมูล
    with st.expander("📊 ฐานข้อมูลกีฬาปัจจุบัน"):
        st.dataframe(df)
else:
    st.warning("กรุณาใส่ลิงก์ SHEET_URL ให้ถูกต้องเพื่อให้ระบบทำงานได้")
