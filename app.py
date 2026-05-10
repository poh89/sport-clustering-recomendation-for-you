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

# --- 💡 ฟังก์ชันวิเคราะห์เหตุผลแบบเจาะลึก (Deep Analysis) ---
def get_detailed_reason(row, user_req):
    reasons = []
    # วิเคราะห์ทีละปัจจัยหลัก
    if row['Intensity'] >= 7 and user_req[0] >= 7:
        reasons.append(f"กีฬานี้มีค่าความเหนื่อยระดับ {row['Intensity']} ซึ่งจะช่วยกระตุ้นการทำงานของหัวใจและหลอดเลือด (Cardiovascular) ได้อย่างดีเยี่ยม ตรงตามที่คุณต้องการเน้นการเบิร์นแคลอรี่")
    
    if row['Social'] >= 7 and user_req[1] >= 7:
        reasons.append(f"ด้วยระดับการเข้าสังคมที่สูงถึง {row['Social']} กีฬานี้จะช่วยให้คุณได้พบปะผู้คนและสร้างมิตรภาพผ่านการทำงานเป็นทีม ซึ่งจะทำให้คุณสนุกกับการออกกำลังกายได้นานขึ้น")
        
    if row['Budget'] <= 4 and user_req[2] <= 4:
        reasons.append(f"ในด้านความคุ้มค่า กีฬานี้มีค่าใช้จ่ายอุปกรณ์ที่ต่ำ (ระดับ {row['Budget']}) ทำให้คุณสามารถเริ่มต้นได้ทันทีโดยไม่มีภาระด้านงบประมาณที่สูงเกินไป")
    
    if row['Flexibility'] >= 7 and user_req[3] >= 7:
        reasons.append(f"คุณจะได้พัฒนาความยืดหยุ่นของเส้นเอ็นและกล้ามเนื้อ ซึ่งช่วยลดโอกาสการบาดเจ็บในชีวิตประจำวันและเพิ่มบุคลิกภาพที่ดี")
        
    if row['Strength'] >= 7 and user_req[4] >= 7:
        reasons.append(f"เน้นการใช้มวลกล้ามเนื้อหลักเพื่อสร้างพละกำลัง (Core Strength) ซึ่งจะช่วยเพิ่มอัตราการเผาผลาญพื้นฐานของร่างกายคุณให้ดีขึ้นในระยะยาว")

    if not reasons:
        return "กีฬานี้มีค่าเฉลี่ยในทุกมิติใกล้เคียงกับภาพรวมที่คุณกำลังมองหามากที่สุด เป็นจุดเริ่มต้นที่สมดุลสำหรับพื้นฐานร่างกายของคุณ"
    
    return " อีกทั้งยัง".join(reasons)

# --- [INTERFACE] ---
st.title("🏆 SportMatch AI Expert")
st.caption("วิเคราะห์กีฬาที่แม่นยำที่สุดด้วยระบบ Machine Learning (Cosine Similarity)")

with st.sidebar:
    st.header("⚙️ ตั้งค่าข้อมูล")
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
            st.success(f"ดึงทักษะเดิมจาก {old_sport} มาคำนวณ")

    with col_b:
        st.subheader("🎯 เป้าหมายที่ต้องการ")
        c1, c2 = st.columns(2)
        with c1:
            u_int = st.select_slider("1. ความเหนื่อย/Cardio", options=range(1, 11), value=5)
            u_soc = st.select_slider("2. การเข้าสังคม/ทีม", options=range(1, 11), value=5)
            u_bud = st.select_slider("3. งบประมาณ (1=ประหยัด)", options=range(1, 11), value=5)
        with c2:
            u_flex = st.select_slider("4. ความยืดหยุ่นร่างกาย", options=range(1, 11), value=5)
            u_str = st.select_slider("5. พละกำลัง/กล้ามเนื้อ", options=range(1, 11), value=5)
        
        button_label = "🚀 แนะนำกีฬาสำหรับคุณ" if not old_sport else "🚀 แนะนำกีฬาอันดับถัดไป"
        run_btn = st.button(button_label, use_container_width=True)

    if run_btn:
        user_req = np.array([u_int, u_soc, u_bud, u_flex, u_str])
        final_vector = (user_req * 0.7) + (bonus_vector * 0.3) if experience == "เคยเล่นกีฬาบางชนิดมาก่อน" else user_req
        
        sim = cosine_similarity([final_vector], df[features])
        df['Score'] = sim[0]
        
        processed_df = df.copy()
        result_title = "🔥 แนะนำกีฬาสำหรับคุณ"
        
        if experience == "เคยเล่นกีฬาบางชนิดมาก่อน":
            processed_df = processed_df[processed_df['Sport'] != old_sport]
            result_title = f"🔥 แนะนำกีฬาอันดับถัดไป (พัฒนาจาก {old_sport})"
            
        recs = processed_df.sort_values(by='Score', ascending=False).head(3)
        
        st.divider()
        st.header(result_title)
        
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            with st.container():
                c_rank, c_info = st.columns([1, 6])
                with c_rank:
                    st.markdown(f"<h1 style='text-align:center; color:#4F8BF9; font-size: 50px;'>#{i}</h1>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(f"### **{row['Sport']}** (ความแม่นยำ {round(row['Score']*100, 1)}%)")
                    
                    # ส่วนของเหตุผลแบบยาว
                    detailed_reason = get_detailed_reason(row, user_req)
                    st.markdown(f"""
                    <div style="background-color: #eef2ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4F8BF9; margin-bottom: 20px;">
                        <strong>ทำไมถึงควรเริ่มกีฬานี้:</strong><br>{detailed_reason}
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
    st.subheader("📊 ตารางคุณสมบัติกีฬาในระบบ")
    st.dataframe(df.drop(columns=['Score'], errors='ignore'), use_container_width=True)
