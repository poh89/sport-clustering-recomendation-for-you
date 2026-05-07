import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. เตรียมข้อมูลกีฬา (Sports Dataset) ---
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

# --- 2. Data Mining: จัดกลุ่มกีฬา (Clustering) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)
cluster_map = {
    0: 'สายชิล/ยืดหยุ่น (Mind & Body)', 
    1: 'สายพลัง/สร้างกล้ามเนื้อ (Strength & Power)', 
    2: 'สายคาร์ดิโอ/สังคม (Cardio & Social)'
}
df['Category'] = df['Cluster'].map(cluster_map)

# --- 3. ส่วนหน้าเว็บ (UI Design) ---
st.set_page_config(page_title="Sport Recommendation AI", layout="wide", page_icon="🎯")

# ตกแต่งหัวข้อหน้าเว็บ
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4F8BF9;'>🎯 ค้นหากีฬาที่ใช่... ในสไตล์คุณ!</h1>
        <p style='color: #666;'>วิเคราะห์ผลด้วย AI (Clustering & Recommendation System)</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# --- [ส่วนที่เพิ่มใหม่: ข้อมูลทั่วไป] ---
st.header("👤 ข้อมูลส่วนตัวของคุณ")
col_info1, col_info2 = st.columns(2)
with col_info1:
    user_gender = st.selectbox("🚻 เพศของคุณ", ["ชาย (Male)", "หญิง (Female)", "อื่นๆ (Other)"])
with col_info2:
    user_age = st.number_input("🎂 อายุของคุณ", min_value=1, max_value=100, value=20)

st.markdown("---")

# --- ส่วนแบบสอบถามความชอบ ---
st.header("📝 คุณชอบออกกำลังกายแบบไหน? (คะแนน 1-10)")
col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    user_intensity = st.slider("🏃‍♂️ ชอบความเหนื่อย/คาร์ดิโอแค่ไหน?", 1, 10, 5)
    user_budget = st.slider("💰 งบประมาณอุปกรณ์ (1=ประหยัด, 10=ทุ่มไม่อั้น)?", 1, 10, 5)
    user_strength = st.slider("💪 เน้นพละกำลัง/สร้างกล้ามเนื้อไหม?", 1, 10, 5)
with col_slider2:
    user_social = st.slider("👥 ชอบเข้าสังคม/เล่นเป็นทีมแค่ไหน?", 1, 10, 5)
    user_flexibility = st.slider("🤸‍♀️ เน้นความยืดหยุ่นของร่างกายไหม?", 1, 10, 5)

# นำข้อมูลเข้าสู่โมเดล
user_profile = np.array([[user_intensity, user_social, user_budget, user_flexibility, user_strength]])

st.markdown("---")

# --- การคำนวณและแสดงผล ---
if st.button("✨ เริ่มวิเคราะห์หาข้อมูล!", use_container_width=True):
    # คำนวณความเหมือนด้วย Cosine Similarity
    similarity_scores = cosine_similarity(user_profile, X)
    df['Match_Score'] = similarity_scores[0]
    recommendations = df.sort_values(by='Match_Score', ascending=False).head(3)
    top_sport = recommendations.iloc[0]['Sport']
    
    # --- [บันทึกข้อมูลลง CSV: เพิ่มเพศและอายุ] ---
    file_name = 'user_responses.csv'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame({
        'Timestamp': [timestamp],
        'Gender': [user_gender],
        'Age': [user_age],
        'Intensity': [user_intensity],
        'Social': [user_social],
        'Budget': [user_budget],
        'Flexibility': [user_flexibility],
        'Strength': [user_strength],
        'Recommended_Sport': [top_sport]
    })
    
    if not os.path.isfile(file_name):
        new_data.to_csv(file_name, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(file_name, mode='a', header=False, index=False, encoding='utf-8-sig')
    
    st.toast('✅ บันทึกข้อมูลเข้าฐานข้อมูลเรียบร้อย!', icon='💾')

    # แสดงผลลัพธ์แบบ Card สวยๆ
    st.markdown("<h2 style='text-align: center;'>🎉 กีฬาที่เหมาะกับคุณ 3 อันดับแรก</h2>", unsafe_allow_html=True)
    res_col1, res_col2, res_col3 = st.columns(3)
    
    result_cols = [res_col1, res_col2, res_col3]
    bg_colors = ['#E8F3FF', '#FFF3E8', '#F3FFE8'] # สีน้ำเงิน, ส้ม, เขียว (อ่อน)
    border_colors = ['#4F8BF9', '#F98F4F', '#4FF98F']
    
    for (i, row), col, bg, border in zip(recommendations.iterrows(), result_cols, bg_colors, border_colors):
        match_percentage = round(row['Match_Score'] * 100, 1)
        with col:
            st.markdown(f"""
                <div style='background-color: {bg}; border: 2px solid {border}; border-radius: 15px; padding: 25px; text-align: center; box-shadow: 2px 4px 10px rgba(0,0,0,0.1);'>
                    <h1 style='font-size: 60px; margin: 0;'>{row['Icon']}</h1>
                    <h2 style='margin: 10px 0;'>{row['Sport']}</h2>
                    <h1 style='color: {border}; margin: 0;'>{match_percentage}%</h1>
                    <p style='color: #666; font-size: 14px;'>กลุ่ม: {row['Category']}</p>
                </div>
                """, unsafe_allow_html=True)

# --- 4. ส่วน Admin Dashboard ---
st.markdown("---")
with st.expander("🔐 ระบบจัดการหลังบ้าน (Admin Only)"):
    if os.path.isfile('user_responses.csv'):
        saved_df = pd.read_csv('user_responses.csv')
        st.write(f"📊 สถิติผู้ใช้งาน: **{len(saved_df)}** คน")
        st.dataframe(saved_df, use_container_width=True, hide_index=True)
        
        # ปุ่มโหลดไฟล์
        csv_report = saved_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 ดาวน์โหลดข้อมูลทั้งหมด (CSV)",
            data=csv_report,
            file_name=f"sport_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        if st.button("🗑️ ล้างข้อมูลทั้งหมด (Reset)"):
            os.remove('user_responses.csv')
            st.rerun()
    else:
        st.info("ยังไม่มีข้อมูลผู้ใช้งานถูกบันทึกไว้")
