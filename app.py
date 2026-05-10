import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- [CONFIGURATION] ---
st.set_page_config(page_title="SportMatch AI Expert", layout="wide", page_icon="🏆")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT-CWhytgN4VrGjp82zbtzbjTrw0HtIqcUtOUuyt8HVwiJbwB9_gRBPt5MRlGUe6AGILR9W2-y8_Inf/pub?output=csv"

features = ['Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']

STATIC_DATA = {
    'Sport': ['วิ่ง', 'ว่ายน้ำ', 'โยคะ', 'ฟุตบอล', 'แบดมินตัน', 'มวยสากล', 'จักรยาน', 'ยกน้ำหนัก', 'เทนนิส', 'ปีนหน้าผา', 'ยิงธนู'],
    'Intensity': [8, 7, 3, 9, 7, 10, 6, 8, 8, 7, 2],
    'Social': [1, 1, 2, 10, 8, 2, 4, 1, 6, 5, 2],
    'Budget': [2, 4, 3, 3, 5, 4, 8, 5, 9, 7, 8],
    'Flexibility': [3, 8, 10, 5, 7, 6, 3, 2, 6, 8, 4],
    'Strength': [4, 6, 5, 6, 4, 8, 5, 10, 6, 9, 6]
}

@st.cache_data(ttl=1)
def load_data(use_google=True):
    if not use_google: return pd.DataFrame(STATIC_DATA)
    try:
        df = pd.read_csv(SHEET_URL).iloc[:, 1:7].dropna()
        df.columns = ['Sport', 'Intensity', 'Social', 'Budget', 'Flexibility', 'Strength']
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(5)
        return df
    except: return pd.DataFrame(STATIC_DATA)

# --- 💡 ฟังก์ชัน AI วิเคราะห์เจาะลึกตาม Slider ของผู้ใช้ ---
def get_advanced_reason(row, user_req):
    analysis = []
    
    # 1. วิเคราะห์ Cardio
    if user_req[0] >= 7:
        analysis.append(f"เนื่องจากคุณเน้นความเหนื่อยระดับสูง ({user_req[0]}/10) กีฬา {row['Sport']} ที่มีค่า Intensity อยู่ที่ {row['Intensity']} จึงเป็นตัวเลือกที่ยอดเยี่ยมในการเสริมสร้างความอึดของกล้ามเนื้อหัวใจ")
    elif user_req[0] <= 3:
        analysis.append(f"จากการที่คุณเลือกความเหนื่อยในระดับต่ำ ({user_req[0]}/10) เราจึงแนะนำ {row['Sport']} ซึ่งเน้นการเคลื่อนไหวที่นุ่มนวล ไม่ทำให้ร่างกายล้าจนเกินไปสำหรับผู้เริ่มต้น")

    # 2. วิเคราะห์ Social
    if user_req[1] >= 7:
        analysis.append(f"ในด้านสังคมที่คุณต้องการสูงถึง {user_req[1]}/10 กีฬานี้จะช่วยตอบโจทย์ด้วยการเป็นกิจกรรมที่เน้นการทำงานเป็นทีมหรือต้องมีคู่เล่น ทำให้คุณไม่รู้สึกโดดเดี่ยว")
    elif user_req[1] <= 3:
        analysis.append(f"คุณเลือกที่ต้องการความเป็นส่วนตัว ({user_req[1]}/10) กีฬานี้จึงเหมาะมากเพราะสามารถฝึกฝนได้ด้วยตนเอง มีสมาธิอยู่กับตัวเองได้เต็มที่")

    # 3. วิเคราะห์ Budget
    if user_req[2] <= 4:
        analysis.append(f"เมื่อพิจารณาจากงบประมาณที่คุณต้องการประหยัด ({user_req[2]}/10) {row['Sport']} เป็นกีฬาที่เข้าถึงง่าย อุปกรณ์น้อยชิ้น ทำให้เริ่มเล่นได้ทันทีโดยไม่มีภาระทางการเงิน")

    # 4. วิเคราะห์ Flexibility & Strength
    if user_req[3] >= 7 or user_req[4] >= 7:
        analysis.append(f"นอกจากนี้ กีฬานี้ยังมีจุดเด่นเรื่องพละกำลังและความยืดหยุ่นที่สอดคล้องกับค่าที่คุณเลือก ซึ่งจะช่วยปรับสมดุลสรีระของคุณให้แข็งแรงขึ้นในระยะยาว")

    # สรุปภาพรวม
    final_text = " ".join(analysis)
    if not final_text:
        final_text = "กีฬานี้ถูกคัดเลือกมาเพราะมีค่าเฉลี่ยในทุกด้านใกล้เคียงกับความต้องการที่คุณระบุไว้ใน Slider มากที่สุด"
    
    return final_text

# --- [INTERFACE] ---
st.title("🏆 SportMatch AI Expert")
st.caption("การวิเคราะห์ความคล้ายคลึงเชิงคณิตศาสตร์ (Cosine Similarity Logic)")

with st.sidebar:
    st.header("⚙️ ตั้งค่าระบบ")
    source = st.radio("แหล่งข้อมูล:", ["Google Form (Real-time)", "ข้อมูลระบบ (Static)"])
    df = load_data(use_google=(source == "Google Form (Real-time)"))

tab1, tab2 = st.tabs(["🎯 วิเคราะห์กีฬา", "📊 ฐานข้อมูล"])

with tab1:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("👤 ข้อมูลพื้นฐาน")
        experience = st.radio("สถานะผู้ใช้งาน:", ["ไม่มีพื้นฐาน", "เคยเล่นกีฬาบางชนิดมาก่อน"])
        old_sport = None
        bonus_vector = np.zeros(5)
        
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            old_sport = st.selectbox("เลือกกีฬาที่คุณเล่นอยู่ในปัจจุบัน:", df['Sport'].unique())
            bonus_vector = df[df['Sport'] == old_sport][features].values[0]
            st.success(f"ดึงฐานทักษะจาก {old_sport}")

    with col_b:
        st.subheader("🎯 เป้าหมายตาม Slider ของคุณ")
        c1, c2 = st.columns(2)
        with c1:
            u_int = st.select_slider("1. ความเหนื่อย/Cardio", options=range(1, 11), value=5)
            u_soc = st.select_slider("2. การเข้าสังคม/ทีม", options=range(1, 11), value=5)
            u_bud = st.select_slider("3. งบประมาณ (1=ประหยัด)", options=range(1, 11), value=5)
        with c2:
            u_flex = st.select_slider("4. ความยืดหยุ่นร่างกาย", options=range(1, 11), value=5)
            u_str = st.select_slider("5. พละกำลัง/กล้ามเนื้อ", options=range(1, 11), value=5)
        
        run_btn = st.button("🚀 เริ่มการวิเคราะห์เชิงลึก", use_container_width=True)

    if run_btn:
        user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if experience == "เคยเล่นกีฬาบางชนิดมาก่อน" else user_req
        
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        
        processed_df = df.copy()
        result_title = "🔥 รายการกีฬาที่แนะนำสำหรับคุณ"
        
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            processed_df = processed_df[processed_df['Sport'] != old_sport]
            result_title = f"🔥 กีฬาอันดับถัดไป (แนะนำต่อเนื่องจาก {old_sport})"
            
        recs = processed_df.sort_values(by='Score', ascending=False).head(3)
        
        st.divider()
        st.header(result_title)
        
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            with st.container():
                c_rank, c_info = st.columns([1, 6])
                with c_rank:
                    st.markdown(f"<h1 style='text-align:center; color:#4F8BF9; font-size: 60px;'>#{i}</h1>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"### **{row['Sport']}** (Matching Score: {round(row['Score']*100, 1)}%)")
                    
                    # --- ส่วนแสดงเหตุผลแบบดึงค่า Slider มาพูด ---
                    detailed_reason = get_advanced_reason(row, user_req)
                    st.markdown(f"""
                    <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 8px solid #4F8BF9; color: #1e3a8a; line-height: 1.6;">
                        <strong>🔍 บทวิเคราะห์จาก AI:</strong><br>{detailed_reason}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # กราฟคุณสมบัติ
                    chart_data = pd.DataFrame({
                        'หัวข้อ': ['Cardio', 'Social', 'Budget', 'Flex', 'Str'],
                        'คะแนน': [row['Intensity'], row['Social'], row['Budget'], row['Flexibility'], row['Strength']]
                    }).set_index('หัวข้อ')
                    st.bar_chart(chart_data.T, height=150)
                st.markdown("---")

with tab2:
    st.subheader("📊 ข้อมูลเชิงลึกในฐานข้อมูล")
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)
