import math

import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title='Dithering', layout='wide')

st.title('Dithering with a Specific Palette')
st.write('Smiliar to color quantisation, dithering reduces the number of colors in an image. '
         'First a color palette is created by clustering the colors in the image. Then the color is dithered. '
         'This method adds pixels to the image to make it look like it has more colors than it actually has.')

st.write("[Color Quantisation](https://en.wikipedia.org/wiki/Color_quantization), "
         "[KMenas](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)"
         "[Dithering](https://en.wikipedia.org/wiki/Dither), ")

st.write("---")

with st.sidebar:
    st.title('Settings')

    image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], key='image')

    dither_n = st.slider('Dithering Matrix Size', min_value=2, max_value=8, value=4, step=1)
    n_colors = st.number_input('Number of colors', min_value=2, max_value=256, value=4, step=1)
    downscale = st.slider('Downscale', min_value=1, max_value=10, value=1, step=1)
    density = st.slider('Density', min_value=1, max_value=100, value=10, step=1)

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
                img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
                q_img, palette = quantize_color_image_unique(img, n_colors)
                sorted_palette = sorted(palette.tolist(), reverse=True)
                d_img = np.full_like(img, sorted_palette[0], dtype=int)
                n = dither_m.shape[0]

                # Create index grids
                x_idx, y_idx = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                dither_idx = dither_m[x_idx % n, y_idx % n]

                for i in range(len(sorted_palette) - 1):
                    current_color = sorted_palette[i]
                    next_color = sorted_palette[i + 1]
                    diff = img - current_color
                    distances = np.linalg.norm(diff, axis=-1)
                    rel_distances = distances / (distances + np.linalg.norm(img - next_color, axis=-1))
                    sigmoided = 1 / (1 + np.exp(-(rel_distances - 0.5) * temperature))
                    dithered = sigmoided > dither_idx
                    d_img[dithered] = next_color

                return d_img


            dither_n = 2 ** dither_n

            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print(img.shape)
            print(dither_n, n_colors, downscale, density)
            img = generate_pixel_art(img, dither_matrix(dither_n), n_colors, downscale, density)
            st.image(img, use_column_width=True)
