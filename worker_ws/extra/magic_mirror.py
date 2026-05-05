import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import math
import pandas as pd
import os

# ==========================================
# 1. MARKET-READY CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Uniform Sizer", page_icon="🔒", layout="centered")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- STANDARD DATA CHARTS ---
SHOULDER_CHART = {90: 23.0, 100: 25.0, 110: 27.0, 120: 29.0, 130: 31.0, 140: 33.5, 150: 36.0, 160: 39.0, 170: 42.0, 180: 44.0}
CHEST_CHART = {90: 54.0, 100: 56.0, 110: 60.0, 120: 64.0, 130: 68.0, 140: 74.0, 150: 78.0, 160: 84.0, 170: 90.0, 180: 96.0}
WAIST_CHART = {90: 50.0, 100: 52.0, 110: 54.0, 120: 58.0, 130: 62.0, 140: 68.0, 150: 72.0, 160: 76.0, 170: 80.0, 180: 84.0}

def get_standard_prediction(height_cm):
    heights = list(SHOULDER_CHART.keys())
    std_s = np.interp(height_cm, heights, list(SHOULDER_CHART.values()))
    std_c = np.interp(height_cm, heights, list(CHEST_CHART.values()))
    std_w = np.interp(height_cm, heights, list(WAIST_CHART.values()))
    return std_s, std_c, std_w

def get_body_proportion_factor(height_cm, age):
    if height_cm > 165: return 0.88
    if age < 6: return 0.83
    if age < 10: return 0.85
    if age < 14: return 0.86
    return 0.88

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# ==========================================
# 2. PRIVACY VISION ENGINE
# ==========================================
def blur_face(image, landmarks, h, w):
    face_x = []
    face_y = []
    for i in range(0, 11): 
        lm = landmarks[i]
        face_x.append(int(lm.x * w))
        face_y.append(int(lm.y * h))
    
    if face_x and face_y:
        min_x, max_x = min(face_x), max(face_x)
        min_y, max_y = min(face_y), max(face_y)
        padding_w = int((max_x - min_x) * 0.8)
        padding_h = int((max_y - min_y) * 1.0)
        x1 = max(0, min_x - padding_w)
        y1 = max(0, min_y - padding_h)
        x2 = min(w, max_x + padding_w)
        y2 = min(h, max_y + int(padding_h/2))
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            roi = cv2.GaussianBlur(roi, (99, 99), 30)
            image[y1:y2, x1:x2] = roi
    return image

def process_video_auto(video_path, user_height_cm, age):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    cap = cv2.VideoCapture(video_path)
    
    ret, check_frame = cap.read()
    if not ret: return [], [], [], None
    h, w, _ = check_frame.shape
    auto_rotate = True if w > h else False
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    m_sh, m_ch, m_waist = [], [], []
    max_width_found = 0
    best_frame = None
    
    TAILOR_FACTOR = 1.12 
    WAIST_FACTOR = 3.55 
    
    current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current += 1
        if current % 2 != 0: continue 
        
        if auto_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_sh, r_sh = lm[11], lm[12]
            l_hip, r_hip = lm[23], lm[24]
            nose, l_heel, r_heel = lm[0], lm[29], lm[30]
            
            frame = blur_face(frame, lm, h, w)
            
            if (l_sh.visibility > 0.6 and r_sh.visibility > 0.6 and l_hip.visibility > 0.6):
                nose_px = (int(nose.x * w), int(nose.y * h))
                heel_px = (int((l_heel.x * w + r_heel.x * w) / 2), int((l_heel.y * h + r_heel.y * h) / 2))
                px_height = calculate_distance(nose_px, heel_px)
                
                if px_height > 100:
                    prop = get_body_proportion_factor(user_height_cm, age)
                    scale = (user_height_cm * prop) / px_height
                    
                    px_sh = calculate_distance((l_sh.x*w, l_sh.y*h), (r_sh.x*w, r_sh.y*h))
                    px_hip = calculate_distance((l_hip.x*w, l_hip.y*h), (r_hip.x*w, r_hip.y*h))
                    
                    sh_cm = px_sh * scale * TAILOR_FACTOR
                    ch_cm = sh_cm * 2.15
                    waist_cm = px_hip * scale * WAIST_FACTOR
                    
                    if 20 < sh_cm < 60:
                        m_sh.append(sh_cm)
                        m_ch.append(ch_cm)
                        m_waist.append(waist_cm)
                        
                        if sh_cm > max_width_found:
                            max_width_found = sh_cm
                            vis_ext = abs(r_hip.x*w - l_hip.x*w) * 0.5
                            cv2.line(frame, nose_px, heel_px, (255, 100, 0), 2)
                            cv2.line(frame, (int(l_sh.x*w), int(l_sh.y*h)), (int(r_sh.x*w), int(r_sh.y*h)), (0,255,0), 4)
                            cv2.line(frame, (int(l_hip.x*w - vis_ext), int(l_hip.y*h)), (int(r_hip.x*w + vis_ext), int(r_hip.y*h)), (0,255,255), 4)
                            best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cap.release()
    return m_sh, m_ch, m_waist, best_frame

# ==========================================
# 3. INTERFACE
# ==========================================
st.title("Smart Uniform Sizer 🔒")
st.caption("Secure AI Measurement Tool")

with st.container():
    c1, c2, c3 = st.columns([2, 1, 2])
    name = c1.text_input("Name", placeholder="Student Name")
    age = c2.number_input("Age", 4, 30, 15)
    phone = c3.text_input("Phone", placeholder="Mobile")
    height = st.number_input("Total Height (cm)", 80, 200, 150)

st.divider()

# --- TUTORIAL SECTION (GIF SUPPORT) ---
st.markdown("### 📹 How to Scan")
with st.expander("Click to see instructions", expanded=True):
    col_gif, col_text = st.columns([1, 2])
    
    with col_gif:
        # Tries to load 'tutorial.gif'. If missing, shows a placeholder.
        if os.path.exists("tutorial.gif"):
            st.image("tutorial.gif", use_container_width=True)
        else:
            st.info("ℹ️ Add 'tutorial.gif' to folder")
            st.markdown("Placeholder: 🧍‍♂️➡️🔄➡️✅")
        
    with col_text:
        st.markdown("""
        **3 Simple Rules:**
        
        ✅ **Full Body Visible:** Head to Toe must be in frame.
        
        ✅ **Fitted Clothes:** Wear a T-shirt and shorts (No jackets).
        
        ✅ **Spin Slowly:** Turn around once (360°) so we see all angles.
        """)

st.divider()

uploaded_file = st.file_uploader("Upload Video", type=['mp4','mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    with st.spinner("Analyzing, please wait..."):
        m_sh, m_ch, m_w, img = process_video_auto(tfile.name, height, age)
    
    if m_sh:
        cam_s = np.percentile(m_sh, 95)
        cam_c = np.percentile(m_ch, 95)
        cam_w = np.percentile(m_w, 95)
        std_s, std_c, std_w = get_standard_prediction(height)
        
        final_chest = (cam_c + std_c) / 2 if abs(cam_c - std_c) < 12 else std_c
        
        # Adaptive Waist Logic
        waist_diff = std_w - cam_w
        if cam_w < std_w:
            if waist_diff < 5.0: final_waist = cam_w 
            else: final_waist = (cam_w * 0.3) + (std_w * 0.7) 
        else:
            final_waist = cam_w

        shirt = int(round(final_chest / 2.54))
        if shirt % 2 != 0: shirt += 1
        pant = int(round(final_waist / 2.54))
        if pant % 2 != 0: pant += 1
        
        conf = "High" if (abs(cam_c - std_c) < 12 and abs(cam_w - std_w) < 20) else "Low"

        st.divider()
        st.success("Analysis Complete!")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(img, caption="Privacy Mode Enabled", use_container_width=True)
        with res_col2:
            st.markdown("### Recommended Size")
            st.markdown(f"<h1 style='color: #4CAF50;'>Shirt: {shirt}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='color: #2196F3;'>Pant: {pant}</h1>", unsafe_allow_html=True)
            
            if conf == "Low": st.warning("Standard sizes used for safety.")

            if st.button("💾 Save Order", type="primary", use_container_width=True):
                if name:
                    new_order = {
                        "Name": [name], "Age": [age], "Phone": [phone], "Height": [height],
                        "Shirt_Size": [shirt], "Pant_Size": [pant], "Confidence": [conf],
                        "Date": [pd.Timestamp.now()]
                    }
                    csv_file = 'shop_orders.csv'
                    df = pd.DataFrame(new_order)
                    if not os.path.isfile(csv_file): df.to_csv(csv_file, index=False)
                    else: df.to_csv(csv_file, mode='a', header=False, index=False)
                    st.toast("Order Saved!", icon="🔒")
                else:
                    st.error("Enter Name first.")
    else:
        st.error("No person detected.")