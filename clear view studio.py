import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import wave
from PIL import Image
from moviepy.editor import VideoFileClip

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Clear View Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"  # Forces Sidebar to stay open
)

# --- 2. NAVIGATION SETUP ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name

# --- 3. FIXED CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF;
        background-color: #0E1117;
    }

    .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-bottom: 5rem;
        text-align: left;
    }

    /* Headings */
    h1, h2, h3 { text-align: left; }
    p { text-align: left; color: #B0B0B0; }

    /* Buttons */
    div.stButton > button {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
        border-radius: 6px;
        height: 3em;
        padding: 0 24px;
        font-weight: 500;
        margin-right: auto;
        display: block;
    }
    div.stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-start;
        border-bottom: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #ff4b4b;
    }

    /* Only hide the hamburger menu and footer, NOT the header (keeps sidebar arrow visible) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# --- 4. HELPER FUNCTIONS ---
def convert_to_bytes(img):
    img_pil = Image.fromarray(img)
    from io import BytesIO
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

def apply_filters(img, mode, bright, contrast):
    out = cv2.convertScaleAbs(img, alpha=contrast, beta=bright)
    if mode == "Black & White":
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif mode == "Sepia":
        img_sepia = np.array(out, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],[0.349, 0.686, 0.168],[0.272, 0.534, 0.131]]))
        out = np.array(np.clip(img_sepia, 0, 255), dtype=np.uint8)
    elif mode == "Sketch":
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        out = cv2.cvtColor(cv2.divide(gray, 255-blur, scale=256), cv2.COLOR_GRAY2RGB)
    elif mode == "Invert":
        out = cv2.bitwise_not(out)
    return out

def detect_ai(img, tool):
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    count = 0
    
    if tool == "Face Detect":
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces: cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count = len(faces)
    elif tool == "ORB Features":
        orb = cv2.ORB_create(nfeatures=500)
        kp = orb.detect(gray, None)
        out = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        count = len(kp)
    elif tool == "Object Detect":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
    elif tool == "Edge Detect":
        edges = cv2.Canny(gray, 100, 200)
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
    return out, count

def render_editor(image_input, key_prefix="img"):
    c_tools, c_img = st.columns([1, 4], gap="medium")
    processed = image_input.copy()

    with c_tools:
        st.markdown("### Settings")
        mode = st.radio("Mode", ["Transform", "Filters", "AI Detect", "Split View", "Grid"], label_visibility="collapsed")
        st.markdown("---")

        if mode == "Transform":
            st.write("Orientation")
            rot = st.selectbox("Rotation", ["0°", "90°", "180°", "270°"], key=f"{key_prefix}_rot")
            fh = st.checkbox("Flip H", key=f"{key_prefix}_fh")
            fv = st.checkbox("Flip V", key=f"{key_prefix}_fv")
            
            if rot == "90°": processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
            elif rot == "180°": processed = cv2.rotate(processed, cv2.ROTATE_180)
            elif rot == "270°": processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if fh: processed = cv2.flip(processed, 1)
            if fv: processed = cv2.flip(processed, 0)

        elif mode == "Filters":
            st.write("Preset")
            filt = st.selectbox("Style", ["None", "Black & White", "Sepia", "Sketch", "Invert"], key=f"{key_prefix}_fil")
            st.write("Tuning")
            br = st.slider("Brightness", -100, 100, 0, key=f"{key_prefix}_br")
            ct = st.slider("Contrast", 0.5, 3.0, 1.0, key=f"{key_prefix}_ct")
            processed = apply_filters(processed, filt, br, ct)

        elif mode == "AI Detect":
            st.write("Tool")
            tool = st.radio("Detector", ["None", "Face Detect", "ORB Features", "Object Detect", "Edge Detect"], key=f"{key_prefix}_ai")
            if tool != "None":
                processed, cnt = detect_ai(processed, tool)
                if cnt > 0: st.success(f"Detected: {cnt}")

        elif mode == "Split View":
            st.write("Layout")
            split = st.selectbox("Type", ["Vert 50/50", "Vert 70/30", "Vert 3-Way", "Horiz 50/50", "Grid 2x2"], key=f"{key_prefix}_sp")

        elif mode == "Grid":
            st.write("Overlay")
            if st.checkbox("Show Grid", key=f"{key_prefix}_gshow"):
                gs = st.slider("Size", 2, 50, 5, key=f"{key_prefix}_gs")
                gc = st.color_picker("Color", "#00FF00", key=f"{key_prefix}_gc")
                # Draw grid
                hc = gc.lstrip('#')
                color = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))[::-1] # RGB to BGR
                h, w, _ = processed.shape
                sx, sy = w//gs, h//gs
                for i in range(1, gs):
                    cv2.line(processed, (i*sx, 0), (i*sx, h), color, 2)
                    cv2.line(processed, (0, i*sy), (w, i*sy), color, 2)

        st.markdown("---")
        st.download_button("Download Image", convert_to_bytes(processed), "result.png", "image/png")

    with c_img:
        # Simple Display Logic
        if mode == "Split View":
            h, w, _ = processed.shape
            if split == "Vert 50/50":
                c1, c2 = st.columns(2)
                c1.image(processed[:, :w//2], caption="Left", use_container_width=True)
                c2.image(processed[:, w//2:], caption="Right", use_container_width=True)
            elif split == "Vert 70/30":
                c1, c2 = st.columns(2)
                mid = int(w*0.7)
                c1.image(processed[:, :mid], caption="L", use_container_width=True)
                c2.image(processed[:, mid:], caption="R", use_container_width=True)
            elif split == "Horiz 50/50":
                c1, c2 = st.columns(2)
                c1.image(processed[:h//2, :], caption="Top", use_container_width=True)
                c2.image(processed[h//2:, :], caption="Bottom", use_container_width=True)
            else:
                st.image(processed, use_container_width=True)
        else:
            st.image(processed, use_container_width=True)

# ==========================================
# 5. SIDEBAR NAVIGATION (FIXED)
# ==========================================
with st.sidebar:
    st.title("Navigation")
    
    # We use a Radio button for the menu, which is more stable than buttons
    # We sync it with the session state
    options = ["Home", "Image Studio", "Video Lab"]
    
    # Determine default index based on current page
    default_ix = 0
    if st.session_state.page == 'image': default_ix = 1
    elif st.session_state.page == 'video': default_ix = 2
    
    selected_page = st.radio("Go to:", options, index=default_ix)
    
    # Logic to change page if radio selection changes
    if selected_page == "Home" and st.session_state.page != 'home':
        st.session_state.page = 'home'
        st.rerun()
    elif selected_page == "Image Studio" and st.session_state.page != 'image':
        st.session_state.page = 'image'
        st.rerun()
    elif selected_page == "Video Lab" and st.session_state.page != 'video':
        st.session_state.page = 'video'
        st.rerun()

# ==========================================
# 6. PAGES
# ==========================================

# --- HOME ---
if st.session_state.page == 'home':
    st.title("CLEAR VIEW STUDIO")
    st.markdown("### Professional Media Analysis")
    st.markdown("---")
    
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown("### Image Studio")
        st.write("Advanced geometry, filters, AI detection for static imagery.")
        # Callback ensures the button works 100% of the time
        st.button("Open Image Studio", on_click=navigate_to, args=('image',))

    with c2:
        st.markdown("### Video Lab")
        st.write("Timeline processing, frame-by-frame inspection, and audio forensics.")
        st.button("Open Video Lab", on_click=navigate_to, args=('video',))

# --- IMAGE ---
elif st.session_state.page == 'image':
    st.markdown("## Image Studio")
    
    c1, c2 = st.columns([1, 4])
    with c1: st.write("**Source**")
    with c2: 
        f = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if f:
        fb = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(fb, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        render_editor(img, "main")
    else:
        st.info("Please upload an image.")

# --- VIDEO ---
elif st.session_state.page == 'video':
    st.markdown("## Video Lab")

    c1, c2 = st.columns([1, 4])
    with c1: st.write("**Source**")
    with c2: 
        v = st.file_uploader("Upload", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")
    
    if v:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(v.read())
        tfile.close()
        
        vf = cv2.VideoCapture(tfile.name)
        frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vf.get(cv2.CAP_PROP_FPS)
        dur = frames/fps if fps else 0
        w, h = int(vf.get(3)), int(vf.get(4))
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time", f"{dur:.2f}s")
        m2.metric("FPS", f"{fps:.2f}")
        m3.metric("Frames", frames)
        m4.metric("Res", f"{w}x{h}")
        
        st.markdown("---")
        st.write("**Timeline**")
        idx = st.slider("Frame", 0, frames-1, 0, label_visibility="collapsed")
        
        vf.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vf.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t1, t2 = st.tabs(["Editor", "Audio"])
            
            with t1: render_editor(frame, "vid")
            with t2:
                if st.button("Analyze Audio"):
                    with st.spinner("Processing..."):
                        try:
                            clip = VideoFileClip(tfile.name)
                            if clip.audio:
                                t_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                                t_wav.close()
                                clip.audio.write_audiofile(t_wav.name, fps=44100, logger=None)
                                
                                with wave.open(t_wav.name, 'rb') as wf:
                                    n = wf.getnframes()
                                    sig = np.frombuffer(wf.readframes(n), dtype=np.int16)
                                    fr = wf.getframerate()
                                    ch = wf.getnchannels()
                                    
                                    sig_f = sig.astype(np.float64)
                                    rms = np.sqrt(np.mean(sig_f**2))
                                    db = 20 * np.log10(rms) if rms>0 else -99
                                    if ch > 1: sig = sig.reshape(-1, ch).mean(axis=1)
                                
                                d1, d2, d3, d4 = st.columns(4)
                                d1.metric("Hz", fr)
                                d2.metric("Ch", ch)
                                d3.metric("RMS", int(rms))
                                d4.metric("dB", f"{db:.1f}")
                                
                                g1, g2 = st.columns(2)
                                g1.line_chart(sig[::1000])
                                fft = np.abs(np.fft.fft(sig[:min(len(sig), 100000)]))
                                g2.area_chart(fft[::100])
                                
                                with open(t_wav.name, "rb") as f:
                                    st.download_button("Download Audio", f.read(), "audio.wav", "audio/wav")
                                
                                clip.close()
                                os.remove(t_wav.name)
                            else: st.warning("No audio.")
                        except Exception as e: st.error(str(e))
        vf.release()
