# Color Quantization & Dithering Demo

This repository contains source code for a Computer Vision & AI homework assignment at HSLU, capped at 20 hours. The project explores color quantization and dithering techniques.

# Color Quantization & Dithering Experiment  

This repository contains source code for a **Computer Vision & AI** homework assignment at **HSLU**. The project explores **color quantization** and **dithering** techniques.  

## 🚀 Streamlit Demo  
Check out the interactive demo:  
🔗 **[Auto-Pixler Streamlit App](https://auto-pixler.streamlit.app)**  

## 📂 Project Overview  
This project experiments with different **color quantization** and **dithering** algorithms to reduce image colors while maintaining visual quality.

### 🎨 Color Quantization

Color quantization reduces the number of colors in an image while trying to keep it looking similar. This is useful for compression, old-school graphics, and printing. It works by grouping similar colors together and replacing them with fewer representative colors.  

[further informations](https://en.wikipedia.org/wiki/Color_quantization)

### 🖼️ Dithering

Dithering is a technique used to fake missing colors by blending pixels of different colors. It creates the illusion of more colors by placing different-colored pixels next to each other. This is common in old video games and black-and-white printing.


[further informations](https://en.wikipedia.org/wiki/Dither)


🔢 K-Means Clustering

K-means is a way to group data into K clusters. In color quantization, it helps find the best K colors to represent an image. It works by:

1. Picking K random colors.
2. Assigning every pixel to the closest color.
3. Updating the colors to be the average of their assigned pixels.
4. Repeating until things stop changing.

[further_information](https://en.wikipedia.org/wiki/K-means_clustering)


## 🏗️ Installation  
If you want to run this code locally:
1. Clone this repository:  
   ```bash
   git clone https://github.com/JakobFenderHSLU/auto-pixler
   cd auto-pixler
   ```
2. Install dependencies:
  ```
  pip install -r requirements.txt
  ```
3. Run the App
  ```bash
  streamlit run app.py
  ```


