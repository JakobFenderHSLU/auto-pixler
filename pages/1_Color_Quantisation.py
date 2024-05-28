import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title='Color Quantisation', layout='wide')

st.title('Color Quantisation')
st.write('Here you can reduce the number of colors in an image. Here the color is quantized per channel. '
         'So either fully red, green, or blue.')

st.write("Further Information: [Color Quantisation](https://en.wikipedia.org/wiki/Color_quantization)")

st.write("---")

with st.sidebar:
    st.title('Settings')

    image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], key='image')

    colors = st.number_input('Number of colors', min_value=2, max_value=256, value=4, step=1)
    black_and_white = st.checkbox('Black and white', value=False)

if image:
    col = st.columns(2)
    with col[0]:
        st.title("Original Image")
        st.image(image, use_column_width=True)
    with col[1]:
        st.title("Processed Image")

        with st.spinner("processing..."):
            def quantize_grey_image(img, n_colors=4):
                palette = np.linspace(0, 255, n_colors).astype(np.uint8)
                ret_img = np.zeros_like(img).astype(np.uint8)
                for i in range(n_colors):
                    ret_img[img >= palette[i]] = palette[i]
                return ret_img


            def quantize_color_image(img, n_colors=4):
                palette = []
                for i in range(n_colors):
                    for j in range(n_colors):
                        for k in range(n_colors):
                            palette.append(
                                [i * 255 // (n_colors - 1), j * 255 // (n_colors - 1), k * 255 // (n_colors - 1)])
                palette = np.array(palette).astype(np.uint8)

                ret_img = np.zeros_like(img).astype(np.uint8)
                for i in range(n_colors ** 3):
                    ret_img[np.all(img >= palette[i], axis=-1)] = palette[i]
                return ret_img


            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if black_and_white:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = quantize_grey_image(img, colors)
            else:
                img = quantize_color_image(img, colors)

            st.image(img, use_column_width=True, output_format="PNG")
