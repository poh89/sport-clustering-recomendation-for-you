import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. จำลองข้อมูลกีฬา (Sports Dataset) ---
@st.cache_data
def load_sports_data():
    data = {
        'Sport': ['วิ่ง (Running)', 'ว่ายน้ำ (Swimming)', 'โยคะ (Yoga)', 'ฟุตบอล (Football)', 
                  'แบดมินตัน (Badminton)', 'มวยสากล (Boxing)', 'ปั่นจักรยาน (Cycling)', 'ยกน้ำหนัก (Weightlifting)'],
        'Intensity': [8, 7, 3, 9, 7, 10, 6, 8], 
        'Social': [1, 1, 2, 10, 8, 2, 4, 1],    
        'Budget': [2, 4, 3, 3, 5, 4, 8, 5],     
        'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2], 
        'Strength': [4, 6, 5, 6, 4, 8, 5, 10]   
    }
    return pd.DataFrame(data)

df = load_sports_data()
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
X = df[features]

# --- 2. ทำ Clustering จัดกลุ่มกีฬา ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

cluster_map = {
    0: 'สายชิล/ยืดหยุ่น (Mind & Body)', 
    1: 'สายพลัง/สร้างกล้ามเนื้อ (Strength & Power)', 
    2: 'สายคาร์ดิโอ/สังคม (Cardio & Social)'
}
df['Category'] = df['Cluster'].map(cluster_map)

# --- 3. ส่วนหน้าเว็บ (Web UI) ---
st.set_page_config(page_title="ค้นหากีฬาที่ใช่สำหรับคุณ!", layout="centered")

st.title("🎯 ค้นหากีฬาที่ใช่... สไตล์คุณ!")
st.write("บอกเราหน่อยว่าคุณชอบอะไร แล้ว AI ของเราจะหากีฬาที่เหมาะสมที่สุดให้ครับ")
st.markdown("---")

# ส่วนรับข้อมูลจากผู้ใช้งาน
st.header("📝 แบบสอบถามความชอบของคุณ (คะแนน 1-10)")

col1, col2 = st.columns(2)
with col1:
    user_intensity = st.slider("🏃‍♂️ ชอบเหนื่อยหอบ/คาร์ดิโอแค่ไหน?", 1, 10, 5)
    user_social = st.slider("👥 ชอบเล่นเป็นกลุ่ม/เจอคนเยอะไหม?", 1, 10, 5)
    user_budget = st.slider("💰 ยอมจ่ายค่าอุปกรณ์/สนามได้แค่ไหน?", 1, 10, 5)
with col2:
    user_flexibility = st.slider("🤸‍♀️ อยากเน้นความยืดหยุ่นของร่างกายไหม?", 1, 10, 5)
    user_strength = st.slider("💪 อยากสร้างกล้ามเนื้อให้แข็งแรงไหม?", 1, 10, 5)

user_profile = np.array([[user_intensity, user_social, user_budget, user_flexibility, user_strength]])

st.markdown("---")

# ปุ่มประมวลผลและบันทึกข้อมูล
if st.button("✨ ค้นหากีฬาของฉัน!", use_container_width=True):
    # คำนวณระบบแนะนำ
    similarity_scores = cosine_similarity(user_profile, X)
    df['Match_Score'] = similarity_scores[0]
    recommendations = df.sort_values(by='Match_Score', ascending=False).head(3)
    
    # ดึงชื่อกีฬาอันดับ 1 มาเป็นคำตอบหลัก
    top_sport = recommendations.iloc[0]['Sport']
    
    # ---------------------------------------------------------
    # ส่วนที่เพิ่มเข้ามา: บันทึกข้อมูลลงไฟล์ CSV (Excel)
    # ---------------------------------------------------------
    file_name = 'user_responses.csv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # สร้าง DataFrame สำหรับข้อมูลใหม่
    new_data = pd.DataFrame({
        'เวลาที่บันทึก': [timestamp],
        'ความเหนื่อย': [user_intensity],
        'สังคม': [user_social],
        'งบประมาณ': [user_budget],
        'ความยืดหยุ่น': [user_flexibility],
        'ความแข็งแรง': [user_strength],
        'กีฬาที่ระบบแนะนำ': [top_sport]
    })
    
    # ตรวจสอบว่ามีไฟล์อยู่แล้วหรือไม่ ถ้าไม่มีให้สร้างใหม่พร้อมหัวตาราง
    if not os.path.isfile(file_name):
        new_data.to_csv(file_name, index=False, encoding='utf-8-sig')
    else:
        # ถ้ามีไฟล์แล้ว ให้ต่อท้ายข้อมูลเดิม (Append)
        new_data.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8-sig')
        
    st.toast('✅ บันทึกข้อมูลลงฐานข้อมูลเรียบร้อยแล้ว!', icon='💾')
    # ---------------------------------------------------------
    
    st.success("🎉 วิเคราะห์เสร็จสิ้น! นี่คือกีฬาที่เหมาะกับคุณ:")
    
    for i, row in recommendations.iterrows():
        match_percentage = round(row['Match_Score'] * 100, 1)
        st.subheader(f"🥇 {row['Sport']} (ความเหมาะสม: {match_percentage}%)")
        st.write(f"**จัดอยู่ในกลุ่ม:** {row['Category']}")
        st.progress(int(match_percentage))
        st.write("") 

# --- 4. ส่วนสำหรับแอดมิน (ดูข้อมูลที่ถูกบันทึก) ---
st.markdown("---")
with st.expander("🔐 สำหรับผู้ดูแลระบบ (Admin: ดูข้อมูลผู้ตอบแบบสอบถาม)"):
    if os.path.isfile('user_responses.csv'):
        saved_df = pd.read_csv('user_responses.csv')
        st.write(f"📊 มีผู้ใช้งานระบบแล้วทั้งหมด: **{len(saved_df)}** คน")
        st.dataframe(saved_df)
        
        # ปุ่มสำหรับเคลียร์ข้อมูล
        if st.button("🗑️ ล้างข้อมูลทั้งหมด"):
            os.remove('user_responses.csv')
            st.rerun()
    else:
        st.info("ยังไม่มีข้อมูลผู้ใช้งานในระบบครับ")