import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Professional", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"
features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

# --- [LOGIC: REASONING ENGINE] ---
def get_dynamic_reason(new_sport, old_sport_row, user_req, is_newbie):
    reasons = []
    
    # --- กรณีที่ 1: สำหรับมือใหม่ (Newbie) ---
    if is_newbie:
        reasons.append(f"สำหรับผู้เริ่มต้น กีฬา {new_sport['Sport']} เป็นตัวเลือกที่น่าสนใจเพราะมีโครงสร้างที่สอดคล้องกับความต้องการของคุณ")
        if user_req[0] >= 7: reasons.append(f"เน้นการฝึกระบบไหลเวียนโลหิตและสร้างความอึด (Cardio) ซึ่งตรงกับที่คุณเลือกไว้ {user_req[0]}/10")
        if user_req[2] <= 4: reasons.append(f"มีค่าใช้จ่ายอุปกรณ์ที่เข้าถึงง่าย (ระดับ {new_sport['Budget']}) เหมาะสำหรับการเริ่มต้นโดยไม่ต้องลงทุนสูง")
        if user_req[3] >= 7: reasons.append(f"ช่วยพัฒนาความอ่อนตัวและลดความตึงเครียดของกล้ามเนื้อสำหรับมือใหม่")
        if user_req[1] <= 3: reasons.append(f"คุณสามารถฝึกซ้อมได้ด้วยตนเอง มีพื้นที่ส่วนตัวในการพัฒนาทักษะตามจังหวะของคุณเอง")
    
    # --- กรณีที่ 2: สำหรับคนมีพื้นฐาน (Switching) ---
    else:
        reasons.append(f"การขยับจาก {old_sport_row['Sport']} มาสู่ {new_sport['Sport']} เป็นการต่อยอดที่ชาญฉลาด")
        
        # เปรียบเทียบความเหนื่อย
        if new_sport['Intensity'] > old_sport_row['Intensity']:
            reasons.append(f"กีฬานี้จะท้าทายระบบหายใจ (Intensity: {new_sport['Intensity']}) ได้เข้มข้นกว่า {old_sport_row['Sport']} เดิมของคุณ")
        else:
            reasons.append(f"ให้คุณรักษาความฟิตได้โดยลดแรงกระแทกหรือความล้าสะสมจากกีฬาเดิม")
            
        # เปรียบเทียบทักษะสังคม
        if new_sport['Social'] > old_sport_row['Social']:
            reasons.append(f"เปิดโอกาสให้คุณได้ฝึกการทำงานเป็นทีม (Teamwork) มากขึ้นกว่าเดิม เพื่อตอบโจทย์สายสังคม {user_req[1]}/10")
        
        # เปรียบเทียบงบประมาณ
        if new_sport['Budget'] < old_sport_row['Budget']:
            reasons.append(f"ช่วยลดภาระค่าใช้จ่ายในการซ่อมบำรุงอุปกรณ์เมื่อเทียบกับกีฬาเดิมที่คุณเล่น")

        # สรุปภาพรวมการพัฒนา
        reasons.append(f"ซึ่งเป็นการอุดช่องว่างทักษะเดิมและเสริมสร้างมวลกล้ามเนื้อในส่วนที่ {old_sport_row['Sport']} อาจยังเข้าไม่ถึง")

    # สุ่มเลือกประโยคมาผสมกันเพื่อความหลากหลาย
    final_text = " อีกทั้งยัง ".join(reasons[:3]) # เลือก 3 ประโยคหลักมาโชว์
    return final_text

# --- [DATA LOADING] ---
@st.cache_data(ttl=1)
def load_data():
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna()
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except:
        return pd.DataFrame({'Sport':['Error'], 'Intensity':[5], 'Social':[5], 'Budget':[5], 'Flexibility':[5], 'Strength':[5]})

# --- [UI INTERFACE] ---
st.title("🏆 SportMatch AI Professional")
st.caption("ระบบวิเคราะห์และแนะนำกีฬาเชิงลึก (Advanced Sports Recommendation Engine)")

df = load_data()

tab1, tab2 = st.tabs(["🎯 วิเคราะห์และแนะนำ", "📊 ตรวจสอบข้อมูล"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("👤 ประวัติผู้ใช้งาน")
        mode = st.radio("เลือกสถานะของคุณ:", ["มือใหม่ (ไม่มีพื้นฐาน)", "มีพื้นฐานกีฬาเดิม"])
        
        old_sport_row = None
        bonus_vector = np.zeros(5)
        
        if mode == "มีพื้นฐานกีฬาเดิม":
            selected_old = st.selectbox("เลือกกีฬาที่คุณเล่นชำนาญ:", df['Sport'].unique())
            old_sport_row = df[df['Sport'] == selected_old].iloc[0]
            bonus_vector = old_sport_row[features].values
            st.success(f"อ้างอิงฐานทักษะจาก: {selected_old}")

    with col_b:
        st.subheader("🎯 กำหนดเป้าหมายใหม่ (Slider)")
        c1, c2 = st.columns(2)
        with c1:
            u_int = st.select_slider("1. ความเหนื่อย/Cardio", options=range(1, 11), value=5)
            u_soc = st.select_slider("2. การเข้าสังคม/ทีม", options=range(1, 11), value=5)
            u_bud = st.select_slider("3. งบประมาณอุปกรณ์", options=range(1, 11), value=5)
        with c2:
            u_flex = st.select_slider("4. ความยืดหยุ่นร่างกาย", options=range(1, 11), value=5)
            u_str = st.select_slider("5. พละกำลัง/กล้ามเนื้อ", options=range(1, 11), value=5)
        
        run_btn = st.button("🚀 เริ่มการวิเคราะห์", use_container_width=True)

    if run_btn:
        user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        is_newbie = (mode == "มือใหม่ (ไม่มีพื้นฐาน)")
        
        # คำนวณ Vector ผสม (ถ้าเป็นมือใหม่ใช้ความต้องการ 100%)
        final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if not is_newbie else user_req
        
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        
        # กรองข้อมูล (ถ้ามีกีฬาเดิม ให้ตัดออกไม่ให้แนะนำซ้ำ)
        processed_df = df.copy()
        if not is_newbie:
            processed_df = processed_df[processed_df['Sport'] != selected_old]
            
        recs = processed_df.sort_values(by='Score', ascending=False).head(3)
        
        st.divider()
        header_text = "🔥 กีฬาที่แนะนำสำหรับคุณ" if is_newbie else f"🔥 กีฬาอันดับถัดไป (แนะนำต่อจาก {selected_old})"
        st.header(header_text)
        
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            with st.container():
                c_rank, c_info = st.columns([1, 6])
                with c_rank:
                    st.markdown(f"<h1 style='text-align:center; color:#4F8BF9;'>#{i}</h1>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"### **{row['Sport']}** (Matching: {round(row['Score']*100, 1)}%)")
                    
                    # --- 💡 แสดงเหตุผลแบบหลากหลายตามสถานะ ---
                    reason = get_dynamic_reason(row, old_sport_row, user_req, is_newbie)
                    st.markdown(f"""
                    <div style="background-color: #f1f5f9; padding: 20px; border-radius: 12px; border-left: 10px solid #4F8BF9; color: #1e293b; font-size: 16px;">
                        <strong>🔍 วิเคราะห์โดย AI:</strong><br>{reason}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # กราฟแสดงผล
                    chart_data = pd.DataFrame({
                        'หัวข้อ': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                        'คะแนน': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                    }).set_index('หัวข้อ')
                    st.bar_chart(chart_data.T, height=160)
                st.markdown("---")

with tab2:
    st.subheader("📊 ฐานข้อมูลกีฬาที่ใช้ในการวิเคราะห์")
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)
