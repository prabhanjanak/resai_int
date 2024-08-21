import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Function to resize the image
def resize_image(image, size=(1080, 720)):
    return cv2.resize(image, size)

# Function to convert the image to grayscale
def gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply a blur effect
def blur_image(image, blur_value):
    return cv2.GaussianBlur(image, (blur_value, blur_value), 0)

# Function to apply edge detection
def edge_image(image):
    return cv2.Canny(image, 100, 200)

# Convert OpenCV image to PIL format
def opencv_to_pil(image):
    if len(image.shape) == 2:  # Grayscale
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Save the processed image
def save_image(image, filename):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    st.download_button(label="Download Processed Image", data=buffer, file_name=filename, mime="image/png")

# Custom CSS to apply the color scheme
def add_custom_css():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #080707;
                color: #C9C9C9;
            }}
            .stButton>button {{
                background-color: #FF4B4B;
                color: #C9C9C9;
                border-radius: 8px;
            }}
            .stTitle {{
                color: #FF4B4B;
            }}
            .stFileUploader {{
                background-color: #502D27;
                border-radius: 8px;
            }}
            .stMarkdown {{
                color: #C9C9C9;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Apply custom CSS
    add_custom_css()

    # Add the logo before the title
    st.image("logo.png", width=150)

    st.title("Streamlit Assignment - Task 3")
    st.markdown("### OpenCV Image Processing Application")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Resize the image
        resized_img = resize_image(img)

        # Display the resized image
        st.image(opencv_to_pil(resized_img), caption='Resized Image (1080x720)', use_column_width=True)

        st.markdown("### Choose an Image Processing Operation")

        # Image processing options
        option = st.selectbox("Select an operation", ("None", "Gray", "Blur", "Edge"))

        processed_img = None
        filename = "processed_image.png"

        if option == "Gray":
            processed_img = gray_image(resized_img)
            st.image(opencv_to_pil(processed_img), caption='Gray Image', use_column_width=True)
            filename = "gray_image.png"

        elif option == "Blur":
            blur_value = st.slider('Blur Level', 1, 49, 5, step=2)  # Ensure odd blur kernel size
            processed_img = blur_image(resized_img, blur_value)
            st.image(opencv_to_pil(processed_img), caption=f'Blurred Image (Level: {blur_value})', use_column_width=True)
            filename = f"blurred_image_level_{blur_value}.png"

        elif option == "Edge":
            processed_img = edge_image(resized_img)
            st.image(opencv_to_pil(processed_img), caption='Edge-detected Image', use_column_width=True)
            filename = "edge_image.png"

        if processed_img is not None:
            save_image(opencv_to_pil(processed_img), filename)

if __name__ == "__main__":
    main()
