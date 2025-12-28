import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Pocket Swing Coach", layout="wide")

st.title("⛳ Pocket Swing Coach")
st.markdown("Upload your golf swing video for instant coaching feedback")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=['mp4', 'mov', 'avi'],
    help="Upload a video of your golf swing from the down-the-line view"
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.subheader("📹 Your Swing")
    st.video(video_path)
    
    if st.button("🔍 Analyze Swing", type="primary"):
        with st.spinner("Analyzing your swing..."):
            st.success("✅ Analysis pipeline ready!")
else:
    st.info("👆 Upload a video to get started")
