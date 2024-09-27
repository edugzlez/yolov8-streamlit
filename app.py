import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Object Detection with YOLO", page_icon="🚀")

github_icon = Image.open("github-mark.png")

def main():
    model = YOLO("yolov8n.pt")
    st.title("Object Detection with YOLO")

    image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)

    with col1:
        det_threshold = st.slider("Score threshold", min_value=0.0, max_value=1.0, value=0.25)

    with col2:
        iou_threshold = st.slider("IoU threshold", min_value=0.0, max_value=1.0, value=0.5)

    if image is not None:
        real_image = Image.open(image)

        container = st.container()
        with st.spinner("Waiting for inference..."):
            result = model(real_image, verbose=False, conf=det_threshold, iou=iou_threshold)[0]

        st.image(result.plot()[:, :, ::-1])

    st.html(
        """
    <footer style="flex-direction: row; justify-content: center; align-items: center; margin-top: 10px; font-size: 0.8em; gap: 10px;">
        <a href="http://github.com/edugzlez/yolov8-streamlit"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub logo" width="24" height="24"></a>
        <span>By <a href="http://github.com/edugzlez">Eduardo González (edugzlez)</a> using <a href="https://github.com/ultralytics/ultralytics">YOLOv8</a></span>
    </footer>
    """
    )

if __name__ == "__main__":
    main()
