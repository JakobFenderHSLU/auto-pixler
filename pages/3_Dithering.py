import base64
from io import BytesIO

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title='Dithering', layout='wide')

st.title('Dithering')
st.write('Smiliar to color quantisation, dithering reduces the number of colors in an image. First the color is  '
         'quantized per channel. So either fully red, green, or blue. Then the color is dithered. This method adds '
         'pixel to the image to make it look like it has more colors than it actually has.')

st.write("Further Information: [Color Quantisation](https://en.wikipedia.org/wiki/Color_quantization), "
         "[Dithering](https://en.wikipedia.org/wiki/Dither)")

st.write("---")

with st.sidebar:
    st.title('Settings')

    image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], key='image')

    dither_n = st.number_input('Dithering Matrix Size', min_value=2, max_value=8, value=4, step=1)
    downscale = st.slider('Downscale', min_value=1, max_value=10, value=1, step=1)

    black_and_white = st.checkbox('Black and white', value=False)

if image:
    col = st.columns(2)
    with col[0]:
        st.title("Original Image")
        st.image(image, use_column_width=True)
    with col[1]:
        st.title("Processed Image")

        with st.spinner("processing..."):
            def dither_matrix(n: int):
                if n == 1:
                    return np.array([[0]])
                else:
                    first = (n ** 2) * dither_matrix(int(n / 2))
                    second = (n ** 2) * dither_matrix(int(n / 2)) + 2
                    third = (n ** 2) * dither_matrix(int(n / 2)) + 3
                    fourth = (n ** 2) * dither_matrix(int(n / 2)) + 1
                    first_col = np.concatenate((first, third), axis=0)
                    second_col = np.concatenate((second, fourth), axis=0)
                    return (1 / n ** 2) * np.concatenate((first_col, second_col), axis=1)


            def ordered_dithering(img: np.array, dither_m: np.array):
                img = img.copy() / 255
                ret_img = np.zeros_like(img)
                n = np.size(dither_m, axis=0)
                for x in range(img.shape[1]):
                    for y in range(img.shape[0]):
                        i = x % n
                        j = y % n
                        if img[y][x] > dither_m[i][j]:
                            ret_img[y][x] = 255
                        else:
                            ret_img[y][x] = 0
                return ret_img


            dither_n = 2 ** dither_n

            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            downscale = 2 ** downscale
            img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))

            if black_and_white:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = ordered_dithering(img, dither_matrix(dither_n))
                img = img.astype(np.uint8)
            else:
                for i in range(3):
                    img[:, :, i] = ordered_dithering(img[:, :, i], dither_matrix(dither_n))

            st.image(img, use_column_width=True, output_format="PNG")
