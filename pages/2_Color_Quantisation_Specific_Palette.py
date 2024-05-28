import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title='Color Quantisation', layout='wide')

st.title('Color Quantisation with a Specific Palette')
st.write('Here you can reduce the number of colors in an image. The color here is first grouped by the KMeans '
         'algorithm. After that, the color is quantized to the nearest color in the palette.')

st.write("Further Information: [Color Quantisation](https://en.wikipedia.org/wiki/Color_quantization), "
         "[KMenas](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)")
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
            def quantize_color_image_unique(img, n_colors=4):
                img = img.copy()
                kmeans = KMeans(n_clusters=n_colors)
                kmeans.fit(img.reshape(-1, 3))
                palette = kmeans.cluster_centers_.astype(np.uint8)

                ret_img = palette[kmeans.labels_].reshape(img.shape).astype(np.uint8)
                return ret_img, palette


            img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            def quantize_color_image_black_and_white(img, n_colors=4):
                img = img.copy()
                kmeans = KMeans(n_clusters=n_colors)
                kmeans.fit(img.reshape(-1, 1))
                palette = kmeans.cluster_centers_.astype(np.uint8)

                ret_img = palette[kmeans.labels_].reshape(img.shape).astype(np.uint8)
                return ret_img, palette


            if black_and_white:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img, color_palette_unique = quantize_color_image_black_and_white(img, colors)
            else:
                img, color_palette_unique = quantize_color_image_unique(img, colors)

            st.image(img, use_column_width=True, output_format="PNG")
