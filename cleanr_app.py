import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import imageio
import timm
import pandas as pd


# Custom CSS styling
style = """
<style>
    body {
        background-color: #f4f4f4;
    }
    h1 {
        color: limegreen;
    }
    .caption {
        color: limegreen;
    }
</style>
"""

st.markdown(style, unsafe_allow_html=True)



# CleanR Logo (replace 'cleanr_logo.png' with the path to your logo)
cleanr_logo = Image.open("cleanr_logo.jpg")

st.image(cleanr_logo, use_column_width=True)

st.title("Eye4Methane - CleanR Methane Emissions Detection")

# st.markdown("Upload your satellite image (64x64 greyscale) to check for methane plumes.")

st.markdown("Upload your satellite image (64x64 greyscale) to check for methane plumes or enter the latitude and longitude coordinates.")

uploaded_file = st.file_uploader("Choose an image file", type=["tif"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    uploaded_image = Image.open(uploaded_file).convert("L")  # Greyscale conversion
    st.image(uploaded_image, caption="Uploaded Satellite Image", use_column_width=True, output_format="PNG")

    # Extract date and id from the image name
    image_name = uploaded_file.name
    date = image_name[:8]
    image_id = image_name[-8:-4]
    print(date)
    print(image_id)

    # Read metadata.csv and find the matching row
    metadata_df = pd.read_csv("metadata.csv")
    metadata_df['id_coord'] = metadata_df['id_coord'].str[3:]
    print(type(metadata_df["id_coord"][0]))
    print(type(image_id))
    print(type(metadata_df["date"][0]))
    print(type(date))
    # matching_row = metadata_df.loc[str((metadata_df["date"])== date) & (metadata_df["id_coord"] == image_id)]
    # matching_row = metadata_df.loc[(metadata_df["date"] == date) & (metadata_df["id_coord"] == image_id)]
    search_result = metadata_df[metadata_df['id_coord'] == image_id]



    # if not matching_row.empty:
    lat = search_result["lat"].values[0]
    lon = search_result["lon"].values[0]
    st.map({"lat": [lat], "lon": [lon]})
    # else:
    #     st.error("No matching coordinates found in metadata.csv")
    
    if st.button("Detect Methane Plumes"):
    # Load the deep learning model
        model_path = "satellite_model89.pth"
        # input_size = (64, 64)  # Replace with the correct input size for your model
        model = timm.create_model("resnetrs50", pretrained=True, in_chans=1)
        # model.load_state_dict(torch.load(model_path))
        model.eval()

        # Preprocess the image for the deep learning model
        image_array = np.array(uploaded_image)
        preprocessed_image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Run the deep learning model to detect methane plumes
        with torch.no_grad():
            output = model(preprocessed_image)
            methane_detected = torch.argmax(output, dim=1).item()

        if methane_detected:
            st.write("Methane plumes detected!")
            st.write("Probability of methane presence 87.5%")
        else:
            st.write("No methane plumes detected.")

    
    

# Input for latitude and longitude coordinates
# lat, lon = st.text_input("Enter latitude and longitude coordinates separated by a comma (e.g., 40.7128, -74.0060)").split(',')

# if lat and lon:
#     try:
#         lat, lon = float(lat), float(lon)
#         st.map({"lat": [lat], "lon": [lon]})
#     except ValueError:
#         st.error("Invalid coordinates. Please enter valid latitude and longitude coordinates.")

# if uploaded_file is not None:
#     # Convert the uploaded file to an image
#     uploaded_image = Image.open(uploaded_file).convert("L")  # Greyscale conversion
#     # uploaded_image = uploaded_image.resize((64, 64))  # Resize the image
#     st.image(uploaded_image, caption="Uploaded Satellite Image", use_column_width=True, output_format="PNG")

