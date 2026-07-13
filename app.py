import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import predict_crop
from utils import predict_fertilizer
from utils import crop_encoder


# PAGE CONFIG


st.set_page_config(
    page_title="Crop and Fertilizer Recommendation System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD CSS


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# SIDEBAR


st.sidebar.title("Crop Recommendation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Crop Recommendation",
        "Fertilizer Recommendation",
        "About"   
    ]
)




# HOME PAGE


if page == "Crop Recommendation":

    st.markdown(
        "<h1 class='title'>Crop Recommendation System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Enter Soil and Weather Parameters</p>",
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0,
            max_value=150,
            value=90
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0,
            max_value=150,
            value=40
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0,
            max_value=250,
            value=40
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=400.0,
            value=120.0
        )

    st.write("")

    predict = st.button("Predict Crop")

    if predict:

        crop, probability = predict_crop(
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        )

        confidence = np.max(probability) * 100

        st.success("Prediction Completed Successfully!")

        st.markdown(
            f"""
            <div class="result">
            Recommended Crop<br><br>
             {crop}<br><br>
            Confidence : {confidence:.2f} %
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        crops = crop_encoder.classes_

        df = pd.DataFrame({
            "Crop": crops,
            "Probability": probability
        })

        df = df.sort_values(
            by="Probability",
            ascending=False
        )

        top3 = df.head(3)

        st.subheader("Top 3 Recommended Crops")

        st.dataframe(
            top3.style.format({
                "Probability": "{:.2%}"
            }),
            use_container_width=True
        )

      

        st.markdown("---")
        st.subheader("Soil Health Analysis")

        col1, col2 = st.columns(2)

        with col1:

            if N < 50:
                st.warning("Nitrogen Level : Low")
            elif N <= 100:
                st.success("Nitrogen Level : Optimal")
            else:
                st.error("Nitrogen Level : High")

            if P < 40:
                st.warning("Phosphorus Level : Low")
            elif P <= 80:
                st.success("Phosphorus Level : Optimal")
            else:
                st.error("Phosphorus Level : High")

            if K < 40:
                st.warning("Potassium Level : Low")
            elif K <= 80:
                st.success("Potassium Level : Optimal")
            else:
                st.error("Potassium Level : High")

        with col2:

            if ph < 5.5:
                st.warning("Soil is Acidic")
            elif ph <= 7.5:
                st.success("Soil pH is Ideal")
            else:
                st.warning("Soil is Alkaline")

            if humidity >= 70:
                st.info("Humidity is Suitable for Most Crops")
            else:
                st.warning("Humidity is Low")

            if rainfall >= 100:
                st.info("Rainfall is Sufficient")
            else:
                st.warning("Rainfall is Low")

        st.markdown("---")
        st.subheader("Weather Summary")

        weather = {
            "Temperature (°C)": [temperature],
            "Humidity (%)": [humidity],
            "Rainfall (mm)": [rainfall]
        }

        weather_df = pd.DataFrame(weather)

        st.dataframe(
            weather_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Input Summary")

        input_df = pd.DataFrame({

            "Feature": [
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "pH",
                "Rainfall"
            ],

            "Value": [
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]

        })

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        result = pd.DataFrame({

            "Recommended Crop": [crop],
            "Confidence (%)": [round(confidence, 2)],
            "Nitrogen": [N],
            "Phosphorus": [P],
            "Potassium": [K],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "pH": [ph],
            "Rainfall": [rainfall]

        })

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction",
            data=csv,
            file_name="crop_prediction.csv",
            mime="text/csv"
        )

        st.markdown("---")



        # fertilizer page

elif page == "Fertilizer Recommendation":

    st.title("Fertilizer Recommendation System")

    st.write(
        "Enter the soil and weather information to get the best fertilizer."
    )

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen",
            0,
            150,
            90,
            key="fert_n"
        )

        P = st.number_input(
            "Phosphorus",
            0,
            150,
            40,
            key="fert_p"
        )

        K = st.number_input(
            "Potassium",
            0,
            250,
            40,
            key="fert_k"
        )

        temperature = st.number_input(
            "Temperature",
            0.0,
            60.0,
            25.0,
            key="fert_temp"
        )

    with col2:

        humidity = st.number_input(
            "Humidity",
            0.0,
            100.0,
            80.0,
            key="fert_hum"
        )

        ph = st.number_input(
            "pH",
            0.0,
            14.0,
            6.5,
            key="fert_ph"
        )

        rainfall = st.number_input(
            "Rainfall",
            0.0,
            400.0,
            120.0,
            key="fert_rain"
        )

        crop = st.selectbox(

            "Crop",

            sorted(crop_encoder.classes_)

        )

    if st.button("Recommend Fertilizer"):

        fertilizer = predict_fertilizer(

            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall,
            crop

        )

        st.success("Recommendation Completed")

        st.markdown(
            f"""
            <div class="result">

            Recommended Fertilizer

            <br><br>

            <b>{fertilizer}</b>

            </div>
            """,
            unsafe_allow_html=True
        )

        # end fertilizer page

elif page == "About":

    st.title("About Crop and Fertilizer Recommendation System")

    st.write(
        """
        This system is designed to assist farmers and agricultural researchers in making smarter decisions about crop selection. By analyzing soil nutrients (N, P, K), pH level, humidity, rainfall, and temperature, it identifies the top three crops best suited for the given conditions. The recommendations are practical, data‑driven, and aimed at improving productivity while reducing uncertainty in farming practices.

        Alongside crop suggestions, the system provides sequential fertilizer recommendations tailored to each crop choice. These fertilizer guidelines ensure balanced nutrient supply, healthier plant growth, and sustainable use of resources. With its simple interface and clear outputs, the system serves as a reliable decision support tool that promotes efficient farming and better yield outcomes.
        """
    )

    st.markdown("---")

    st.subheader("📌 Input Features")

    feature_df = pd.DataFrame({

        "Feature": [
            "Nitrogen (N)",
            "Phosphorus (P)",
            "Potassium (K)",
            "Temperature",
            "Humidity",
            "pH",
            "Rainfall"
        ],

        "Description": [
            "Nitrogen content in soil",
            "Phosphorus content in soil",
            "Potassium content in soil",
            "Temperature (°C)",
            "Relative Humidity (%)",
            "Soil pH",
            "Rainfall (mm)"
        ]

    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Dataset Information")

    dataset_info = pd.DataFrame({

        "Property": [
            "Samples",
            "Input Features",
            "Target Variable",
            "Machine Learning Algorithm"
        ],

        "Value": [
            "2200",
            "7",
            "Crop, Fertilizer",
            "Random Forest Classifier"
        ]

    })

    st.table(dataset_info)

    


   

st.markdown(
    """
<div class="footer">

Crop Recommendation System

Developed using Machine Learning | Streamlit

</div>
""",
    unsafe_allow_html=True
)