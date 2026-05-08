import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. ข้อมูลกีฬา ---
@st.cache_data
def load_sports_data():
    data = {
        'Sport': ['วิ่ง (Running)', 'ว่ายน้ำ (Swimming)', 'โยคะ (Yoga)', 'ฟุตบอล (Football)', 
                  'แบดมินตัน (Badminton)', 'มวยสากล (Boxing)', 'ปั่นจักรยาน (Cycling)', 'ยกน้ำหนัก (Weightlifting)'],
        'Icon': ['🏃‍♂️', '🏊‍♂️', '🧘‍♀️', '⚽', '🏸', '🥊', '🚴‍♂️', '💪'],
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

# --- 2. ทำ Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
cluster_map = {0: 'สายชิล/ยืดหยุ่น', 1: 'สายพลัง/สร้างกล้ามเนื้อ', 2: 'สายคาร์ดิโอ/สังคม'}
df['Category'] = df['Cluster'].map(cluster_map)

# --- 3. หน้าเว็บ ---
st.set_page_config(page_title="Sport AI Recommendation", layout="wide", page_icon="🎯")

st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>🎯 ค้นหากีฬาที่ใช่... ในสไตล์คุณ!</h1>", unsafe_allow_html=True)
st.markdown("---")

st.header("👤 ข้อมูลพื้นฐาน")
cp1, cp2 = st.columns(2)
with cp1:
    user_gender = st.selectbox("🚻 เพศของคุณ", ["ชาย (Male)", "หญิง (Female)", "อื่นๆ (Other)"])
with cp2:
    user_age = st.number_input("🎂 อายุของคุณ", min_value=1, max_value=100, value=20)

st.header("📝 คุณชอบออกกำลังกายแบบไหน? (คะแนน 1-10)")
col1, col2 = st.columns(2)
with col1:
    u_intensity = st.slider("🏃‍♂️ ชอบความเหนื่อย/คาร์ดิโอแค่ไหน?", 1, 10, 5)
    u_budget = st.slider("💰 งบประมาณอุปกรณ์?", 1, 10, 5)
    u_strength = st.slider("💪 เน้นพละกำลังกล้ามเนื้อไหม?", 1, 10, 5)
with col2:
    u_social = st.slider("👥 ชอบเข้าสังคม/เล่นเป็นทีมแค่ไหน?", 1, 10, 5)
    u_flexibility = st.slider("🤸‍♀️ เน้นความยืดหยุ่นร่างกายไหม?", 1, 10, 5)

user_profile = np.array([[u_intensity, u_social, u_budget, u_flexibility, u_strength]])

if st.button("✨ วิเคราะห์กีฬาที่เหมาะกับฉัน!", use_container_width=True):
    sim_scores = cosine_similarity(user_profile, X)
    df['Match_Score'] = sim_scores[0]
    recs = df.sort_values(by='Match_Score', ascending=False).head(3)
    
    file_name = 'user_responses.csv'
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({
        'Timestamp': [ts], 'Gender': [user_gender], 'Age': [user_age],
        'Intensity': [u_intensity], 'Social': [u_social], 'Budget': [u_budget],
        'Flexibility': [u_flexibility], 'Strength': [u_strength],
        'Result': [recs.iloc[0]['Sport']]
    })
    
    if not os.path.isfile(file_name):
        new_data.to_csv(file_name, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8-sig')
    
    st.success("🎉 วิเคราะห์เสร็จสิ้น!")
    res_cols = st.columns(3)
    for (i, row), col in zip(recs.iterrows(), res_cols):
        with col:
            st.markdown(f"""
                <div style='background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center;'>
                    <h1>{row['Icon']}</h1>
                    <h3>{row['Sport']}</h3>
                    <h2 style='color: #4F8BF9;'>{round(row['Match_Score']*100, 1)}%</h2>
                </div>
            """, unsafe_allow_html=True)

# --- ส่วนที่แก้ไขปัญหา Error ---
st.markdown("---")
with st.expander("🔐 ระบบจัดการหลังบ้าน (Admin Only)"):
    file_path = 'user_responses.csv'
    if os.path.isfile(file_path):
        try:
            # พยายามอ่านไฟล์
            saved_df = pd.read_csv(file_path)
            st.write(f"📊 มีผู้ใช้งานระบบแล้ว: **{len(saved_df)}** คน")
            st.dataframe(saved_df, use_container_width=True)
            
            if st.button("🗑️ ล้างข้อมูลทั้งหมด"):
                os.remove(file_path)
                st.rerun()
        except Exception as e:
            # ถ้าอ่านไฟล์ไม่ได้ (เพราะโครงสร้างไฟล์เก่าขัดกับโค้ดใหม่)
            st.error("⚠️ ตรวจพบไฟล์ข้อมูลรุ่นเก่าที่ไม่รองรับ หรือไฟล์เสียหาย")
            if st.button("🚨 ลบไฟล์ที่เสียและเริ่มใหม่ (Fix Error)"):
                os.remove(file_path)
                st.rerun()
    else:
        st.info("ยังไม่มีข้อมูลในระบบ")
