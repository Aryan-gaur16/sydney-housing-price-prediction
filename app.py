import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sydney Property Valuation")
MODEL_PATH = "housing_price_model.pkl"
MIN_SENSIBLE_PRICE = 100_000  # Nothing can be blow this value to make predictions accurate

@st.cache_resource
def load_trained_pipeline():
    return joblib.load(MODEL_PATH)

trained_pipeline = load_trained_pipeline()

st.title("Sydney Property Valuation")
st.markdown(
    """
    This website is used to predict the property prices in Blacktown,
    Ashfield and Cronulla using a linear regression model trained across
    103 listings taken from Domain.com.au .
    """
)

st.divider()
st.subheader("Property Details")

col_left, col_right = st.columns(2)

with col_left:
    chosen_suburb = st.radio("Suburb", options=["Blacktown", "Ashfield", "Cronulla"])
    chosen_type = st.radio("Property Type", options=["House", "Apartment", "Townhouse"])
    num_beds = st.slider("Bedrooms", 1, 10, 2) # the default value is set to 2
    num_baths = st.slider("Bathrooms", 1, 6, 2)# the default value is set to 2

with col_right:
    num_parking = st.slider("Parking Spaces", 0, 6, 1) # the default value is set to 1
    lot_size = st.number_input("Land Size in m² (set to 0 if apartment/unknown)", min_value=0, value=0, step=10)
    market_days = st.number_input("Estimated Days on Market", min_value=0, value=30, step=5)
    method_of_sale = st.selectbox("Sale Method", ["private treaty", "auction", "prior to auction"])

st.divider()

def build_feature_row():
    ratio = num_beds / num_baths if num_baths > 0 else float(num_beds)
    land_flag = int(lot_size > 0)
    land_value = lot_size if lot_size > 0 else None

    return pd.DataFrame([{
        "suburb": chosen_suburb,
        "property_type": chosen_type,
        "bedrooms": num_beds,
        "bathrooms": num_baths,
        "parking": num_parking,
        "land_size": land_value,
        "has_land_size": land_flag,
        "days_on_market": market_days,
        "sale_method": method_of_sale,
        "bed_bath_ratio": ratio
    }])

if st.button("Estimate Sale Price", type="primary"):
    feature_row = build_feature_row()
    estimated_price = trained_pipeline.predict(feature_row)[0]

    if estimated_price < MIN_SENSIBLE_PRICE:
        st.error(
            "This combination will produce an unreliable estimate. "
            "Our evaluation found this happens mostly for "
            "apartments with no recorded land size, since land size had to be estimated "
            "rather than measured."
        )
    else:
        st.metric(label="Estimated Sale Price", value=f"${estimated_price:,.0f}")

    with st.expander("Why might this estimate be wrong?"):
        st.write(
            """
            Based on the model evaluation, predictions tend to be less reliable for:
            - Apartments where land size wasn't recorded on the original listing
            - Properties with unusually high or low bedroom/bathroom counts for their suburb
            - Properties where price is driven by condition, renovation, or views.
              None of which this model has access to
            """
        )