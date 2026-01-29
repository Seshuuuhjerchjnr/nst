import os
import streamlit as st
import numpy as np
import PIL.Image

# Reduce TF logging noise BEFORE importing tensorflow-based code
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from nst import NeuralStyleTransfer

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

# ----------- Blue Theme (Improved Contrast) -----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
}

/* Force all text to light color */
html, body, [class*="css"]  {
    color: #E6F7FF !important;
}

/* Headers */
h1, h2, h3 {
    color: #A8E6FF !important;
}

/* Sliders + labels */
label, .stSlider, .stSelectbox {
    color: #E6F7FF !important;
}

/* Buttons */
.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("Neural Style Transfer App")

# ----------- Speed Presets -----------
speed_mode = st.select_slider(
    "Processing Speed",
    options=["Faster", "Medium", "Slower"],
    value="Faster"
)

speed_config = {
    "Faster": (1, 50),
    "Medium": (1, 100),
    "Slower": (3, 200)
}

epochs, steps = speed_config[speed_mode]

# ----------- Style Intensity -----------
style_level = st.select_slider(
    "Style Intensity",
    options=["Low", "Medium", "High"],
    value="Medium"
)

style_map = {
    "Low": 1e-3,
    "Medium": 1e-2,
    "High": 5e-2
}

style_weight = style_map[style_level]

# ----------- Content Preservation -----------
content_level = st.select_slider(
    "Content Preservation",
    options=["Low", "Medium", "High"],
    value="Medium"
)

content_map = {
    "Low": 5e3,
    "Medium": 1e4,
    "High": 2e4
}

content_weight = content_map[content_level]


image_size = st.slider(
    "Image Resolution (Higher = Better Quality, Slower)",
    128, 512, 256
)

uploaded_content = st.file_uploader("Upload Content Image")
uploaded_style = st.file_uploader("Upload Style Image")

#caching improves performance by avoiding reloading the model
@st.cache_resource
def load_model(style_weight, content_weight, image_size):
    return NeuralStyleTransfer(
        style_weight=style_weight,
        content_weight=content_weight,
        max_dim=image_size
    )

if st.button("Stylize Image"):

    if uploaded_content and uploaded_style:

        os.makedirs("temp", exist_ok=True)

        content_path = f"temp/{uploaded_content.name}"
        style_path = f"temp/{uploaded_style.name}"

        with open(content_path, "wb") as f:
            f.write(uploaded_content.getbuffer())

        with open(style_path, "wb") as f:
            f.write(uploaded_style.getbuffer())

        nst = load_model(style_weight, content_weight, image_size)

        with st.spinner("Applying artistic transformation..."):
            result, _ = nst.run(
                content_path,
                style_path,
                epochs=epochs,
                steps_per_epoch=steps
            )

        result_image = np.array(result[0] * 255, dtype=np.uint8)
        result_pil = PIL.Image.fromarray(result_image)

        st.image(result_pil, caption="Stylized Output", width=600)

        # download button
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Stylized Image",
            data=byte_im,
            file_name="stylized.png",
            mime="image/png"
        )
