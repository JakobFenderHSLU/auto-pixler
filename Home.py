import streamlit as st

st.set_page_config(page_title='Home')
st.title('Color Quantization and Dithering')

""" This app was created for the Computer Vision & AI course at the University of Applied Sciences Lucerne.
The app allows you to perform color quantization and dithering on an image of your choice."""

""" The code can be found on [GitHub](https://github.com/JakobFenderHSLU/auto-pixler)"""

st.write("---")

st.write("### Color Quantization")
st.write("- [Color Quantization](/1_Color_Quantisation.py)")
st.write("- [Color Quantization with a Specific Palette](/2_Color_Quantisation_Specific_Palette.py)")

st.write("### Dithering")
st.write("- [Dithering](/3_Dithering.py)")
st.write("- [Dithering with a Specific Palette](/4_Dithering_Specific_Palette.py)")



