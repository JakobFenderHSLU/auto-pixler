import math

import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title='Dithering', layout='wide')

st.title('Dithering')
st.write('Here you can reduce the number of colors in an image.')

st.write("---")

with st.sidebar:
    st.title('Settings')

    image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], key='image')

    dither_n = st.slider('Dithering Matrix Size', min_value=2, max_value=8, value=4, step=1)
    n_colors = st.number_input('Number of colors', min_value=2, max_value=256, value=4, step=1)
    downscale = st.slider('Downscale', min_value=1, max_value=10, value=1, step=1)
    density = st.slider('Density', min_value=1, max_value=100, value=20, step=1)

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


            def quantize_color_image_unique(img, n_colors=4):
                img = img.copy()
                kmeans = KMeans(n_clusters=n_colors)
                kmeans.fit(img.reshape(-1, 3))
                palette = kmeans.cluster_centers_.astype(np.uint8)

                ret_img = palette[kmeans.labels_].reshape(img.shape).astype(np.uint8)
                return ret_img, palette


            def generate_pixel_art(img: np.array, dither_m: np.array, n_colors: int, downscale: int = 1,
                                   temperature: float = 20):
                img = img.copy()
                img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
                q_img, palette = quantize_color_image_unique(img, n_colors)
                sorted_palette = sorted(palette.tolist())
                sorted_palette.reverse()
                d_img = np.ones_like(img).astype('int')

                d_img *= sorted_palette[0]

                for i in range(len(sorted_palette) - 1):
                    sub_img = np.zeros_like(img)
                    current_color = sorted_palette[i]
                    next_color = sorted_palette[i + 1]

                    n = np.size(dither_m, axis=0)
                    for x in range(img.shape[1]):
                        for y in range(img.shape[0]):
                            i = x % n
                            j = y % n

                            distance_current_color = math.sqrt(sum(img[y][x] - current_color) ** 2)
                            distance_next_color = math.sqrt(sum(img[y][x] - next_color) ** 2)
                            rel_dist_next_color = distance_current_color / (
                                    distance_current_color + distance_next_color)
                            sigmoided = 1 / (1 + math.exp(-(rel_dist_next_color - 0.5) * temperature))

                            if sigmoided > dither_m[i][j]:
                                sub_img[y][x] = 255
                            else:
                                sub_img[y][x] = 0
                    # where one multiply with next color and overlay over d_img
                    d_img = np.where(sub_img == 255, next_color, d_img)

                return d_img


            dither_n = 2 ** dither_n
            downscale = 2 ** downscale

            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print(img.shape)
            print(dither_n, n_colors, downscale, density)
            img = generate_pixel_art(img, dither_matrix(dither_n), n_colors, downscale, density)
            st.image(img, use_column_width=True)
