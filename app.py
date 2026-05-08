import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. เชื่อมต่อ Google Sheet ---
# ใส่ลิงก์ที่คุณส่งมาให้ผมเมื่อกี้ลงในนี้ครับ
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

@st.cache_data(ttl=1)
def load_data():
    try:
        df_sheet = pd.read_csv(SHEET_URL)
        # ป้องกัน Error ถ้าคอลัมน์เกินมา: เอาแค่ 7 คอลัมน์แรก (Timestamp + 6 คำถาม)
        df_sheet = df_sheet.iloc[:, :7]
        # ตั้งชื่อคอลัมน์ใหม่ให้ระบบเข้าใจ
        df_sheet.columns = ['Timestamp', 'Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        return df_sheet
    except Exception as e:
        st.error(f"⚠️ ไม่สามารถดึงข้อมูลได้: {e}")
        return pd.DataFrame()

df = load_data()
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- 2. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="SportMatch AI", layout="wide", page_icon="🏆")
st.title("🏆 SportMatch AI: ค้นหากีฬาจากฐานข้อมูลจริง")
st.write("วิเคราะห์ข้อมูลจาก Google Form แบบ Real-time")
st.markdown("---")

if not df.empty:
    # เมนูเลือก Case (ข้างซ้าย)
    mode = st.sidebar.selectbox("🎯 เลือกโหมดการแนะนำ", ["สำหรับมือใหม่ (Newbie)", "สำหรับนักกีฬา (Switching)"])

    if mode == "สำหรับมือใหม่ (Newbie)":
        st.header("🌟 แนะนำกีฬาเริ่มต้นที่เหมาะกับสไตล์คุณ")
        col1, col2 = st.columns(2)
        with col1:
            u_intensity = st.slider("ระดับความเหนื่อยที่ต้องการ", 1, 10, 5)
            u_social = st.slider("ความชอบเข้าสังคม/เล่นเป็นทีม", 1, 10, 5)
            u_budget = st.slider("งบประมาณอุปกรณ์ (1=ประหยัด)", 1, 10, 5)
        with col2:
            u_flex = st.slider("ความยืดหยุ่นร่างกาย", 1, 10, 5)
            u_strength = st.slider("ความแรง/พละกำลัง", 1, 10, 5)

        if st.button("✨ เริ่มวิเคราะห์หาข้อมูล", use_container_width=True):
            user_vector = np.array([[u_intensity, u_social, u_budget, u_flex, u_strength]])
            sim = cosine_similarity(user_vector, df[features])
            df['Score'] = sim[0]
            res = df.sort_values(by='Score', ascending=False).head(3)
            
            st.balloons()
            st.success("กีฬาที่ AI แนะนำคุณ:")
            for i, row in res.iterrows():
                st.write(f"### 🥇 {row['Sport']} (ความเหมาะสม {round(row['Score']*100, 1)}%)")

    else:
        st.header("🔄 ค้นพบความท้าทายใหม่ (จากทักษะที่คุณมี)")
        if not df['Sport'].empty:
            current_sport = st.selectbox("ตอนนี้คุณเล่นกีฬาอะไรอยู่?", df['Sport'].unique())
            
            if st.button("🚀 ค้นหากีฬาที่คล้ายกัน", use_container_width=True):
                target = df[df['Sport'] == current_sport][features].values
                others = df[df['Sport'] != current_sport].copy()
                
                if not others.empty:
                    sim = cosine_similarity(target, others[features])
                    others['Score'] = sim[0]
                    res = others.sort_values(by='Score', ascending=False).head(3)
                    
                    st.info(f"เพราะคุณเล่น {current_sport} เราเลยคิดว่าคุณน่าจะชอบ:")
                    for i, row in res.iterrows():
                        st.write(f"### 🔥 {row['Sport']} (ทักษะใกล้เคียงกัน {round(row['Score']*100, 1)}%)")
                else:
                    st.warning("ต้องมีกีฬาในฐานข้อมูลมากกว่า 1 อย่างเพื่อเปรียบเทียบครับ")

    # ส่วน Admin: ดูข้อมูลจาก Google Sheet
    st.markdown("---")
    with st.expander("📊 ดูฐานข้อมูลที่ดึงจาก Google Sheet (Admin)"):
        st.write(f"จำนวนข้อมูลปัจจุบัน: {len(df)} รายการ")
        st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ ฐานข้อมูลว่างเปล่า กรุณากรอกข้อมูลใน Google Form ก่อนครับ")
