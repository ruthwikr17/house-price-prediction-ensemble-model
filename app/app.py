import os
import sys

# Add the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
from src.ensemble_predict import PricePredictor
import plotly.express as px
import json  # Import json for handling LLM response
import requests  # Import requests for API calls
import re  # Import regex for post-processing

from dotenv import load_dotenv

load_dotenv()


# Define base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Indian_Real_Estate_Clean_Data.csv")


# --- LLM API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="House Price Predictor", layout="wide")


# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the real estate dataset and handles outliers."""
    try:
        df = pd.read_csv(DATA_PATH)

        # --- Outlier Handling ---
        # Removed: df = df[df['Total_Rooms'] >= df['BHK']] # This line was incorrectly filtering out valid data.

        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df = df.dropna(subset=["Price"])  # Drop NaN prices
        df = df[df["Price"] <= 100000000]  # 10 Crore

        df["BHK"] = pd.to_numeric(df["BHK"], errors="coerce")
        df = df.dropna(subset=["BHK"])  # Ensure BHK is not NaN after conversion
        df = df[df["BHK"] <= 8]

        return df
    except FileNotFoundError:
        st.error(
            f"Error: Dataset not found at {DATA_PATH}. Please ensure the 'data' directory and 'Indian_Real_Estate_Clean_Data.csv' exist."
        )
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()


df = load_data()

# --- CRITICAL DEBUGGING STEP (Retained for future diagnostics) ---
print(
    "DEBUG (Data Load Check): Verifying if expected properties are in loaded DataFrame 'df'."
)
expected_property_title_part = "4 BHK Villa for sale in Shadnagar, Hyderabad"
expected_price_inr = 6000000.0  # 60 Lakhs
expected_bhk = 4
expected_property_type = "Villa"
expected_city = "Hyderabad"

check_df = df[
    (
        df["Property Title"].str.contains(
            expected_property_title_part, case=False, na=False
        )
    )
    & (df["Price"] == expected_price_inr)
    & (pd.to_numeric(df["BHK"], errors="coerce") == expected_bhk)
    & (df["property_type"].str.lower().str.strip() == expected_property_type.lower())
    & (df["city"].str.lower() == expected_city.lower())
]

if not check_df.empty:
    print(
        f"DEBUG (Data Load Check): Found {len(check_df)} rows matching expected property."
    )
    print("DEBUG (Data Load Check): Sample of matched rows:")
    # Removed 'Total_Rooms' from debug print as it's not core to data existence check
    print(check_df[["Property Title", "Price", "BHK", "property_type", "city"]].head())
else:
    print(
        "DEBUG (Data Load Check): DID NOT FIND expected property (4 BHK Villa in Shadnagar, Hyderabad, 60.0 Lakhs) in loaded DataFrame 'df'."
    )
    print(
        "DEBUG (Data Load Check): This suggests the issue is still in initial data loading or cleaning."
    )
# --- END CRITICAL DEBUGGING STEP ---


# Instantiate the PricePredictor class
predictor = PricePredictor()

# --- Extract unique values for dropdowns and suggestions ---
unique_cities = sorted(df["city"].dropna().unique())
property_types = sorted(df["property_type"].dropna().unique())
balcony_options = sorted(df["Balcony"].dropna().unique())
unique_bhk_options = sorted(df["BHK"].dropna().unique())
unique_bhk_options = [int(bhk) for bhk in unique_bhk_options if pd.notna(bhk)]
unique_bhk_options = sorted(list(set(unique_bhk_options)))


# --- Pre-processing for general neighborhoods suggestions ---
def extract_neighborhood(full_location_string):
    parts = str(full_location_string).split(",")
    if len(parts) >= 2:
        candidate = parts[-2].strip()
        if candidate.lower() in [c.lower() for c in unique_cities]:
            if len(parts) >= 3:
                return parts[-3].strip()
        return candidate
    return full_location_string.strip()


unique_neighborhoods = sorted(
    list(set(df["Location"].dropna().apply(extract_neighborhood)))
)


# --- Helper Function for Price Formatting for Display (Always in Lakhs) ---
def format_price_for_display(val):
    """
    Formats a numerical price into a human-readable string consistently in Lakhs.
    e.g., 6000000.0 -> "60 Lakhs", 15000000.0 -> "150 Lakhs"
    """
    price_in_lakhs = val / 1e5
    if price_in_lakhs == int(price_in_lakhs):
        return f"₹{int(price_in_lakhs)} Lakhs"
    else:
        return f"₹{price_in_lakhs:.2f} Lakhs"


# --- Helper Function for Price Formatting for Prediction Result (Crores/Lakhs) ---
def format_price(val):
    """
    Formats a numerical price into a human-readable string (Lakhs, Crores)
    for the single prediction result display.
    """
    if val >= 1e7:
        return f"₹{val / 1e7:.2f} Cr"
    elif val >= 1e5:
        return f"₹{val / 1e5:.2f} Lakhs"
    else:
        return f"₹{val:,.0f}"


# --- LLM Interaction Function ---
def get_llm_filtered_data(user_query_text):
    """
    Sends user query to LLM to extract structured filters and applies them to the dataset.
    """
    # Simplified prompt to avoid token limits
    prompt = f"""
    Extract real estate filters from the user query into a JSON object.
    Your response MUST be ONLY a JSON object. Do not include any other text.
    Set keys to null if not present or cannot be confidently extracted.

    Keys: 'city', 'location', 'property_type', 'bhk', 'min_price_lakhs', 'max_price_lakhs', 'min_area_sqft', 'max_area_sqft'.
    Prices (min/max_price_lakhs) should be in lakhs (e.g., 80 for 80 Lakhs, 150 for 1.5 Crore).
    BHK should be an integer.

    Example: "Find 4BHK houses in Hyderabad under 80 Lakhs" -> {{ "city": "Hyderabad", "property_type": "House", "bhk": 4, "max_price_lakhs": 80 }}

    User Query: "{user_query_text}"
    """

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}  # Uses the GEMINI_API_KEY variable

    # Request a JSON schema response from the LLM
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "city": {"type": "STRING", "nullable": True},
                    "location": {"type": "STRING", "nullable": True},
                    "property_type": {"type": "STRING", "nullable": True},
                    "bhk": {"type": "INTEGER", "nullable": True},
                    "min_price_lakhs": {"type": "NUMBER", "nullable": True},
                    "max_price_lakhs": {"type": "NUMBER", "nullable": True},
                    "min_area_sqft": {"type": "NUMBER", "nullable": True},
                    "max_area_sqft": {"type": "NUMBER", "nullable": True},
                },
            },
        },
    }

    print(f"DEBUG (LLM Query): User Query: '{user_query_text}'")

    try:
        response = requests.post(
            GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload)
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        llm_result = response.json()
        print(f"DEBUG (LLM Response): Raw LLM Result: {llm_result}")

        if llm_result and llm_result.get("candidates"):
            llm_json_string = llm_result["candidates"][0]["content"]["parts"][0]["text"]
            extracted_filters = json.loads(llm_json_string)
            print(f"DEBUG (LLM Filters): Extracted Filters: {extracted_filters}")

            # --- Post-processing to ensure price/BHK filters are not missed if implied by query ---
            user_query_lower = user_query_text.lower()

            # Attempt to extract BHK from query if LLM missed it
            if extracted_filters.get("bhk") is None:
                bhk_match = re.search(r"(\d+)\s*bhk", user_query_lower)
                if bhk_match:
                    try:
                        extracted_filters["bhk"] = int(bhk_match.group(1))
                        print(
                            f"DEBUG (Post-process): Extracted BHK {extracted_filters['bhk']} from query."
                        )
                    except ValueError:
                        pass  # Ignore if conversion fails

            # Attempt to extract max_price_lakhs from query if LLM missed it
            if extracted_filters.get("max_price_lakhs") is None:
                price_match_lakh = re.search(
                    r"(?:under|below|less than)\s*(\d+(?:\.\d+)?)\s*lakh",
                    user_query_lower,
                )
                price_match_crore = re.search(
                    r"(?:under|below|less than)\s*(\d+(?:\.\d+)?)\s*crore",
                    user_query_lower,
                )

                if price_match_lakh:
                    try:
                        price_value = float(price_match_lakh.group(1))
                        extracted_filters["max_price_lakhs"] = price_value
                        print(
                            f"DEBUG (Post-process): Extracted max_price_lakhs {extracted_filters['max_price_lakhs']} from query."
                        )
                    except ValueError:
                        pass
                elif price_match_crore:
                    try:
                        price_value = float(price_match_crore.group(1))
                        extracted_filters["max_price_lakhs"] = (
                            price_value * 100
                        )  # Convert crores to lakhs
                        print(
                            f"DEBUG (Post-process): Extracted max_price_lakhs {extracted_filters['max_price_lakhs']} (from Crores) from query."
                        )
                    except ValueError:
                        pass

            # Attempt to extract min_price_lakhs from query if LLM missed it
            if extracted_filters.get("min_price_lakhs") is None:
                price_match_lakh = re.search(
                    r"(?:above|more than)\s*(\d+(?:\.\d+)?)\s*lakh", user_query_lower
                )
                price_match_crore = re.search(
                    r"(?:above|more than)\s*(\d+(?:\.\d+)?)\s*crore", user_query_lower
                )

                if price_match_lakh:
                    try:
                        price_value = float(price_match_lakh.group(1))
                        extracted_filters["min_price_lakhs"] = price_value
                        print(
                            f"DEBUG (Post-process): Extracted min_price_lakhs {extracted_filters['min_price_lakhs']} from query."
                        )
                    except ValueError:
                        pass
                elif price_match_crore:
                    try:
                        price_value = float(price_match_crore.group(1))
                        extracted_filters["min_price_lakhs"] = (
                            price_value * 100
                        )  # Convert crores to lakhs
                        print(
                            f"DEBUG (Post-process): Extracted min_price_lakhs {extracted_filters['min_price_lakhs']} (from Crores) from query."
                        )
                    except ValueError:
                        pass
            # --- END Post-processing ---

            # Apply filters to the main DataFrame
            filtered_data = df.copy()
            print(f"DEBUG (Filtering): Initial data rows: {len(filtered_data)}")

            if extracted_filters.get("city"):
                print(
                    f"DEBUG (Filtering): Applying city filter: '{extracted_filters['city']}'"
                )
                filtered_data = filtered_data[
                    filtered_data["city"].str.lower()
                    == extracted_filters["city"].lower()
                ]
                print(
                    f"DEBUG (Filtering): After city filter: {len(filtered_data)} rows remaining."
                )

            # For location, check if the general location is contained in the detailed 'Location' column
            if extracted_filters.get("location"):
                print(
                    f"DEBUG (Filtering): Applying location filter: '{extracted_filters['location']}'"
                )
                filtered_data = filtered_data[
                    filtered_data["Location"]
                    .str.lower()
                    .str.contains(extracted_filters["location"].lower(), na=False)
                ]
                print(
                    f"DEBUG (Filtering): After location filter: {len(filtered_data)} rows remaining."
                )

            if extracted_filters.get("property_type"):
                # Validate property_type against actual allowed types in the dataset and ensure robust matching
                extracted_pt_lower = (
                    extracted_filters["property_type"].lower().strip()
                )  # Strip whitespace here
                print(
                    f"DEBUG (Filtering): Applying property_type filter: '{extracted_filters['property_type']}' (normalized: '{extracted_pt_lower}')"
                )
                if extracted_pt_lower in [
                    pt.lower().strip() for pt in property_types
                ]:  # Compare with stripped values
                    filtered_data = filtered_data[
                        filtered_data["property_type"].str.lower().str.strip()
                        == extracted_pt_lower  # Filter with stripped values
                    ]
                else:
                    print(
                        f"DEBUG (Filtering): Invalid property_type extracted: '{extracted_filters['property_type']}'. Skipping filter."
                    )
                    # Optionally, you might want to inform the user about invalid property type
                print(
                    f"DEBUG (Filtering): After property_type filter: {len(filtered_data)} rows remaining."
                )

            if (
                extracted_filters.get("bhk") is not None
            ):  # Check for None explicitly as 0 is a valid BHK
                print(
                    f"DEBUG (Filtering): Applying BHK filter: '{extracted_filters['bhk']}'"
                )
                # Ensure 'BHK' column is numeric for comparison
                filtered_data = filtered_data[
                    pd.to_numeric(filtered_data["BHK"], errors="coerce")
                    == extracted_filters["bhk"]
                ]
                print(
                    f"DEBUG (Filtering): After BHK filter: {len(filtered_data)} rows remaining."
                )

            # Convert lakhs to actual price for filtering
            if extracted_filters.get("min_price_lakhs") is not None:
                price_val_inr = extracted_filters["min_price_lakhs"] * 1e5
                print(
                    f"DEBUG (Filtering): Applying min_price_lakhs filter: '{extracted_filters['min_price_lakhs']}' ({price_val_inr:.0f} INR)"
                )
                # Print the prices of the remaining data before this filter
                print(
                    f"DEBUG (Filtering): Prices BEFORE min_price_lakhs filter: {filtered_data['Price'].tolist()}"
                )
                filtered_data = filtered_data[filtered_data["Price"] >= price_val_inr]
                print(
                    f"DEBUG (Filtering): After min_price_lakhs filter: {len(filtered_data)} rows remaining."
                )
            if extracted_filters.get("max_price_lakhs") is not None:
                price_val_inr = extracted_filters["max_price_lakhs"] * 1e5
                print(
                    f"DEBUG (Filtering): Applying max_price_lakhs filter: '{extracted_filters['max_price_lakhs']}' ({price_val_inr:.0f} INR)"
                )
                # Print the prices of the remaining data before this filter
                print(
                    f"DEBUG (Filtering): Prices BEFORE max_price_lakhs filter: {filtered_data['Price'].tolist()}"
                )
                filtered_data = filtered_data[filtered_data["Price"] <= price_val_inr]
                print(
                    f"DEBUG (Filtering): After max_price_lakhs filter: {len(filtered_data)} rows remaining."
                )

            if extracted_filters.get("min_area_sqft") is not None:
                print(
                    f"DEBUG (Filtering): Applying min_area_sqft filter: '{extracted_filters['min_area_sqft']}'"
                )
                filtered_data = filtered_data[
                    filtered_data["Total_Area(SQFT)"]
                    >= extracted_filters["min_area_sqft"]
                ]
                print(
                    f"DEBUG (Filtering): After min_area_sqft filter: {len(filtered_data)} rows remaining."
                )
            if extracted_filters.get("max_area_sqft") is not None:
                print(
                    f"DEBUG (Filtering): Applying max_area_sqft filter: '{extracted_filters['max_area_sqft']}'"
                )
                filtered_data = filtered_data[
                    filtered_data["Total_Area(SQFT)"]
                    <= extracted_filters["max_area_sqft"]
                ]
                print(
                    f"DEBUG (Filtering): After max_area_sqft filter: {len(filtered_data)} rows remaining."
                )

            return filtered_data, extracted_filters
        else:
            print("DEBUG (LLM Candidates): No candidates found in LLM response.")
            return pd.DataFrame(), {}  # Return empty DataFrame if no candidates
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to AI service: {e}")
        print(f"ERROR (API Request): {e}")
        return pd.DataFrame(), {}
    except json.JSONDecodeError as e:
        st.error(
            f"Error parsing AI response: {e}. Raw response might be: {response.text if 'response' in locals() else 'N/A'}"
        )
        print(
            f"ERROR (JSON Decode): {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}"
        )
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"An unexpected error occurred during AI search: {e}")
        print(f"ERROR (Unexpected): {e}")
        return pd.DataFrame(), {}


# --- Navigation Bar ---
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Predict House Price", "Visualize and Filter", "AI Search"],
        icons=["house", "bar-chart", "robot"],
        default_index=0,
    )

# --- Predict the House Page ---
if selected == "Predict House Price":
    st.title("Predict House Prices using Advanced ML Models")
    st.markdown("---")

    # Initialize session state for form visibility and input values
    if "show_prediction_form" not in st.session_state:
        st.session_state.show_prediction_form = False
    if "location_input_value" not in st.session_state:
        st.session_state.location_input_value = "All"  # Default location to "All"
    if "city_input_value" not in st.session_state:
        st.session_state.city_input_value = "All"  # Default city to "All"
    # Ensure location options are stored in session state to maintain state across reruns
    if "prediction_location_options" not in st.session_state:
        st.session_state.prediction_location_options = ["All"] + unique_neighborhoods

    # Initial welcome message and button
    if not st.session_state.show_prediction_form:
        st.markdown(
            """
            <h2 style='text-align:center; color: #1a73e8; font-family: "Inter", sans-serif; font-size: 2.8em;'>
                Get Accurate House Price Predictions based on Available Data
            </h2>
            <br>
            """,
            unsafe_allow_html=True,
        )
        col_btn1, col_btn2, col_btn3 = st.columns([1, 0.5, 1])
        with col_btn2:
            if st.button("Check Out", key="try_now_button"):
                st.session_state.show_prediction_form = True
                st.rerun()
    else:
        # --- Prediction Form Layout (Half-screen width) ---
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("Enter Property Details")

            # Callback function to handle city selection/change
            def update_city_and_location_options_callback():
                # When city selectbox changes, update city_input_value
                st.session_state.city_input_value = (
                    st.session_state.city_selectbox_field
                )
                # Reset location input and its selectbox when city changes
                st.session_state.location_input_value = "All"
                # The location selectbox key needs to be reset too to reflect default_location_index change
                if "location_selectbox_field" in st.session_state:
                    # This will force the selectbox to re-render with "All" selected
                    st.session_state["location_selectbox_field"] = "All"

                # Dynamically update prediction_location_options in session state
                current_city = st.session_state.city_input_value.strip()
                if current_city and current_city != "All":
                    city_specific_locations_df = df[
                        df["city"].str.lower() == current_city.lower()
                    ]["Location"].dropna()
                    st.session_state.prediction_location_options = sorted(
                        list(
                            set(city_specific_locations_df.apply(extract_neighborhood))
                        )
                    )
                else:
                    temp_all_locations = (
                        df["Location"].dropna().apply(extract_neighborhood)
                    )
                    st.session_state.prediction_location_options = sorted(
                        list(set(temp_all_locations))
                    )

                st.session_state.prediction_location_options = [
                    "All"
                ] + st.session_state.prediction_location_options

            # --- DYNAMIC CITY SELECTBOX ---
            # Determine the default index for the city selectbox
            default_city_index = 0
            try:
                if st.session_state.city_input_value in (["All"] + unique_cities):
                    default_city_index = (["All"] + unique_cities).index(
                        st.session_state.city_input_value
                    )
            except ValueError:
                st.session_state.city_input_value = (
                    "All"  # Reset to "All" if current value invalid
                )
                default_city_index = 0

            selected_city = st.selectbox(
                "City",
                options=["All"] + unique_cities,  # Show all cities
                key="city_selectbox_field",  # Unique key for the selectbox
                index=default_city_index,
                placeholder="Select or type a city",
                on_change=update_city_and_location_options_callback,  # Trigger callback on change
            )

            # --- END DYNAMIC CITY SELECTBOX ---

            # --- DYNAMIC LOCATION SELECTBOX FOR PREDICTION SECTION ---
            # Ensure prediction_location_options is always populated correctly on initial load/rerun
            # This handles cases where state might be loaded before callback fires or on direct page load
            current_city_for_location_filter = st.session_state.city_input_value.strip()
            if (
                current_city_for_location_filter
                and current_city_for_location_filter != "All"
            ):
                city_specific_locations_df = df[
                    df["city"].str.lower() == current_city_for_location_filter.lower()
                ]["Location"].dropna()
                temp_location_options = sorted(
                    list(set(city_specific_locations_df.apply(extract_neighborhood)))
                )
            else:
                temp_location_options = unique_neighborhoods
            st.session_state.prediction_location_options = [
                "All"
            ] + temp_location_options

            # Get the current list of location options from session state
            current_location_options = st.session_state.prediction_location_options

            # Determine the default index for the location selectbox
            default_location_index = 0
            try:
                if st.session_state.location_input_value in current_location_options:
                    default_location_index = current_location_options.index(
                        st.session_state.location_input_value
                    )
            except ValueError:
                st.session_state.location_input_value = (
                    "All"  # Reset to "All" or a sensible default
                )
                default_location_index = 0

            # Location Selectbox
            selected_location = st.selectbox(
                "Location (e.g., Kondapur, Jubilee Hills)",
                options=current_location_options,
                key="location_selectbox_field",  # Unique key for the selectbox
                index=default_location_index,
                placeholder="Select or type a neighborhood",
            )

            # Update session state with the selected value
            if selected_location != st.session_state.location_input_value:
                st.session_state.location_input_value = selected_location
                # No rerun needed as the change in selectbox will trigger a rerun automatically

            # Display a message if no specific location options are found for the selected city
            if (
                st.session_state.city_input_value
                and st.session_state.city_input_value != "All"
                and len(st.session_state.prediction_location_options) <= 1
            ):  # Only "All" would be 1
                st.info(
                    f"No specific neighborhoods found in '{st.session_state.city_input_value}'. You can proceed with 'All' or consider typing a precise location if known."
                )

            # --- END DYNAMIC LOCATION SELECTBOX FOR PREDICTION SECTION ---

            area = st.number_input(
                "Total Area (SQFT)", min_value=100, max_value=50000, step=50, value=1200
            )
            bhk = st.number_input("BHK", min_value=1, max_value=8, step=1, value=2)
            property_type = st.selectbox(
                "Property Type",
                options=property_types,
                index=property_types.index("Apartment")
                if "Apartment" in property_types
                else 0,
            )
            balcony = st.selectbox(
                "Balcony",
                options=balcony_options,
                index=balcony_options.index("Yes") if "Yes" in balcony_options else 0,
            )

            if st.button("Get Prediction", key="submit_prediction"):
                location_val = st.session_state.location_input_value.strip()
                city_val = st.session_state.city_input_value.strip()

                if not city_val or city_val == "All":
                    st.warning("Please select a specific City to get a prediction.")
                else:
                    predicted_prices = []
                    city_filtered_df = df[df["city"].str.lower() == city_val.lower()]

                    if location_val == "All" or not location_val:
                        locations_to_predict = (
                            city_filtered_df["Location"].dropna().unique()
                        )
                        prediction_type_info = f"average price across all available properties in {city_val}"
                    else:
                        locations_to_predict = (
                            city_filtered_df[
                                city_filtered_df["Location"]
                                .str.lower()
                                .str.contains(location_val.lower(), na=False)
                            ]["Location"]
                            .dropna()
                            .unique()
                        )
                        prediction_type_info = f"average price for properties in '{location_val}' within {city_val}"

                    if len(locations_to_predict) == 0:
                        st.warning(
                            f"No properties found for '{location_val}' in '{city_val}' within the dataset to average predictions. Please refine your inputs or try different values."
                        )
                    else:
                        progress_text = (
                            f"Calculating {prediction_type_info}... 0% complete."
                        )
                        my_bar = st.progress(0, text=progress_text)
                        for i, granular_loc in enumerate(locations_to_predict):
                            input_dict = {
                                "Location": granular_loc,
                                "Total_Area(SQFT)": area,
                                "Total_Rooms": bhk + 1,
                                "Balcony": balcony,
                                "city": city_val,
                                "property_type": property_type,
                                "BHK": bhk,
                            }
                            try:
                                result = predictor.predict(input_dict)
                                xgb_price = result.get("predicted_price", None)

                                if xgb_price is not None:
                                    predicted_prices.append(xgb_price)
                                progress_value = (i + 1) / len(locations_to_predict)
                                my_bar.progress(
                                    progress_value,
                                    text=f"Calculating {prediction_type_info}... {int(progress_value * 100)}% complete.",
                                )
                            except Exception as e:
                                st.error(
                                    f"Warning: Could not predict for granular location '{granular_loc}' in {city_val}. Error: {e}"
                                )
                        my_bar.empty()

                    if predicted_prices:
                        final_price = np.mean(predicted_prices)
                        st.markdown("---")
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #e6ffe6;
                                padding: 15px;
                                border-radius: 10px;
                                border: 1px solid #4CAF50;
                                margin-top: 15px;
                                text-align: center; /* Center all inline and inline-block content */
                            ">
                                <h3 style="color: #2e7d32; margin-bottom: 5px; text-align: center;">
                                    💰 Predicted House Price ({"Average" if len(predicted_prices) > 1 else "Estimated"}):
                                </h3>
                                <h2 style="color: #1a73e8; font-size: 3em; margin-top: 0px; text-align: center;">
                                    {format_price(final_price)}
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if len(predicted_prices) > 1:
                            st.info(
                                f"💡 Showing {prediction_type_info} based on {len(predicted_prices)} properties."
                            )
                    else:
                        st.warning(
                            "Could not generate a prediction based on the provided inputs. Please check your City/Location and try again."
                        )


# --- Visualize Dataset Page ---

elif selected == "Visualize and Filter":
    st.title("📊 Explore and Filter Dataset")
    st.markdown("---")

    st.subheader("Filter Dataset Records")

    col_filters1, col_filters2, col_filters3 = st.columns(3)

    with col_filters1:
        filter_city = st.selectbox(
            "Filter by City", options=["All"] + unique_cities, key="filter_city"
        )
        filter_property_type = st.selectbox(
            "Filter by Property Type",
            options=["All"] + property_types,
            key="filter_property_type",
        )
    with col_filters2:
        if filter_city != "All":
            city_specific_locations_df = df[df["city"] == filter_city][
                "Location"
            ].dropna()
            city_specific_neighborhoods = sorted(
                list(set(city_specific_locations_df.apply(extract_neighborhood)))
            )
            location_options = ["All"] + city_specific_neighborhoods
        else:
            location_options = ["All"] + unique_neighborhoods

        filter_location = st.selectbox(
            "Filter by Location (Neighborhood)",
            options=location_options,
            key="filter_location",
        )
        filter_bhk = st.selectbox(
            "Filter by BHK", options=["All"] + unique_bhk_options, key="filter_bhk"
        )
    with col_filters3:
        max_allowed_price_lakhs = float(df["Price"].max() / 1e5)
        min_price, max_price = st.slider(
            "Filter by Price Range (Lakhs)",
            min_value=float(df["Price"].min() / 1e5),
            max_value=max_allowed_price_lakhs,
            value=(float(df["Price"].min() / 1e5), max_allowed_price_lakhs),
            step=100.0,
        )
        min_area, max_area = st.slider(
            "Filter by Area Range (SQFT)",
            min_value=float(df["Total_Area(SQFT)"].min()),
            max_value=float(df["Total_Area(SQFT)"].max()),
            value=(
                float(df["Total_Area(SQFT)"].min()),
                float(df["Total_Area(SQFT)"].max()),
            ),
            step=50.0,
        )

    filtered_df = df.copy()

    if filter_city != "All":
        filtered_df = filtered_df[filtered_df["city"] == filter_city]
    if filter_property_type != "All":
        filtered_df = filtered_df[filtered_df["property_type"] == filter_property_type]
    if filter_location != "All":
        filtered_df = filtered_df[
            filtered_df["Location"].apply(
                lambda x: filter_location.lower() in str(x).lower()
            )
        ]
    if filter_bhk != "All":
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df["BHK"], errors="coerce") == filter_bhk
        ]

    filtered_df = filtered_df[
        (filtered_df["Price"] >= min_price * 1e5)
        & (filtered_df["Price"] <= max_price * 1e5)
    ]
    filtered_df = filtered_df[
        (filtered_df["Total_Area(SQFT)"] >= min_area)
        & (filtered_df["Total_Area(SQFT)"] <= max_area)
    ]

    st.subheader("Filtered Dataset Records (10-15 rows)")
    if not filtered_df.empty:
        # Create a display copy to modify for UI without affecting underlying data
        display_df = filtered_df.copy()

        # Remove 'Total_Rooms' column for display
        if "Total_Rooms" in display_df.columns:
            display_df = display_df.drop(columns=["Total_Rooms"])

        # Apply price formatting
        display_df["Price"] = display_df["Price"].apply(format_price_for_display)

        st.dataframe(
            display_df.head(15).reset_index(drop=True).rename(index=lambda x: x + 1)
        )
        st.info(
            f"Displaying top {min(len(filtered_df), 15)} of {len(filtered_df)} matching records."
        )
    else:
        st.warning("No records found matching your filter criteria.")

    st.markdown("---")

    st.subheader("Visualizations for Filtered Data")
    if not filtered_df.empty:
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            fig1 = px.histogram(
                filtered_df,
                x="Price",
                nbins=50,
                title="1. Price Distribution",
                labels={"Price": "Price (INR)"},
                hover_data={"Price": ":,.0f"},
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.scatter(
                filtered_df,
                x="Total_Area(SQFT)",
                y="Price",
                color="BHK",
                title="2. Area (SQFT) vs Price",
                labels={
                    "Total_Area(SQFT)": "Total Area (SQFT)",
                    "Price": "Price (INR)",
                },
                hover_data={"Price": ":,.0f", "Total_Area(SQFT)": True, "BHK": True},
            )
            st.plotly_chart(fig2, use_container_width=True)

            if len(filtered_df["city"].unique()) > 1:
                fig_box_city_filtered = px.box(
                    filtered_df,
                    x="city",
                    y="Price",
                    title="3. Price Distribution by City (Filtered Data)",
                    labels={"city": "City", "Price": "Price (INR)"},
                    hover_data={"Price": ":,.0f"},
                    color="city",
                )
                st.plotly_chart(fig_box_city_filtered, use_container_width=True)
            else:
                st.info(
                    "3. Box Plot by City: Not enough cities in filtered data for meaningful comparison."
                )

            counts_df_prop_filtered = (
                filtered_df["property_type"].value_counts().reset_index()
            )
            counts_df_prop_filtered.columns = ["Property_Type_Category", "Count"]
            fig_count_prop_filtered = px.bar(
                counts_df_prop_filtered,
                x="Property_Type_Category",
                y="Count",
                title="4. Property Count by Type (Filtered Data)",
                labels={"Property_Type_Category": "Property Type", "Count": "Count"},
                color="Property_Type_Category",
            )
            st.plotly_chart(fig_count_prop_filtered, use_container_width=True)

        with plot_col2:
            avg_price_by_type_filtered = (
                filtered_df.groupby("property_type")["Price"].mean().reset_index()
            )
            fig3 = px.bar(
                avg_price_by_type_filtered.sort_values(by="Price", ascending=False),
                x="property_type",
                y="Price",
                title="5. Average Price by Property Type (Filtered Data)",
                labels={
                    "property_type": "Property Type",
                    "Price": "Average Price (INR)",
                },
                hover_data={"Price": ":,.0f"},
                color="property_type",
            )
            st.plotly_chart(fig3, use_container_width=True)

            fig_rooms_filtered = px.scatter(
                filtered_df,
                x="Total_Rooms",
                y="Price",
                color="BHK",
                title="6. Total Rooms vs Price (Filtered Data)",
                labels={"Total_Rooms": "Total Rooms", "Price": "Price (INR)"},
                hover_data={"Price": ":,.0f", "Total_Rooms": True, "BHK": True},
            )
            st.plotly_chart(fig_rooms_filtered, use_container_width=True)

            if len(filtered_df["Balcony"].unique()) > 1:
                fig_box_balcony_filtered = px.box(
                    filtered_df,
                    x="Balcony",
                    y="Price",
                    title="7. Price Distribution by Balcony (Filtered Data)",
                    labels={"Balcony": "Balcony Available", "Price": "Price (INR)"},
                    hover_data={"Price": ":,.0f"},
                    color="Balcony",
                )
                st.plotly_chart(fig_box_balcony_filtered, use_container_width=True)
            else:
                st.info(
                    "7. Box Plot by Balcony: Not enough variation in 'Balcony' for filtered data."
                )

            counts_df_city_filtered = filtered_df["city"].value_counts().reset_index()
            counts_df_city_filtered.columns = ["City_Category", "Count"]
            fig_count_city_filtered = px.bar(
                counts_df_city_filtered,
                x="City_Category",
                y="Count",
                title="8. Property Count by City (Filtered Data)",
                labels={"City_Category": "City", "Count": "Count"},
                color="City_Category",
            )
            st.plotly_chart(fig_count_city_filtered, use_container_width=True)

    else:
        st.info("No records found matching your filter criteria.")

    st.markdown("---")

    st.subheader("Visualizations for Entire Dataset")

    plot_col3, plot_col4 = st.columns(2)

    with plot_col3:
        fig_all1 = px.histogram(
            df,
            x="Price",
            nbins=50,
            title="1. Overall Price Distribution",
            labels={"Price": "Price (INR)"},
            hover_data={"Price": ":,.0f"},
        )
        st.plotly_chart(fig_all1, use_container_width=True)

        fig_all2 = px.scatter(
            df,
            x="Total_Area(SQFT)",
            y="Price",
            color="property_type",
            title="2. Overall Area (SQFT) vs Price by Property Type",
            labels={"Total_Area(SQFT)": "Total Area (SQFT)", "Price": "Price (INR)"},
            hover_data={
                "Price": ":,.0f",
                "Total_Area(SQFT)": True,
                "property_type": True,
            },
        )
        st.plotly_chart(fig_all2, use_container_width=True)

        fig_box_city_all = px.box(
            df,
            x="city",
            y="Price",
            title="3. Overall Price Distribution by City",
            labels={"city": "City", "Price": "Price (INR)"},
            hover_data={"Price": ":,.0f"},
            color="city",
        )
        st.plotly_chart(fig_box_city_all, use_container_width=True)

        counts_df_prop_all = df["property_type"].value_counts().reset_index()
        counts_df_prop_all.columns = ["Property_Type_Category", "Count"]
        fig_count_prop_all = px.bar(
            counts_df_prop_all,
            x="Property_Type_Category",
            y="Count",
            title="4. Overall Property Count by Type",
            labels={"Property_Type_Category": "Property Type", "Count": "Count"},
            color="Property_Type_Category",
        )
        st.plotly_chart(fig_count_prop_all, use_container_width=True)

    with plot_col4:
        avg_price_by_city_all = df.groupby("city")["Price"].mean().reset_index()
        fig_all3 = px.bar(
            avg_price_by_city_all.sort_values(by="Price", ascending=False),
            x="city",
            y="Price",
            title="5. Overall Average Price by City",
            labels={"city": "City", "Price": "Average Price (INR)"},
            hover_data={"Price": ":,.0f"},
            color="city",
        )
        st.plotly_chart(fig_all3, use_container_width=True)

        fig_box_bhk_all = px.box(
            df,
            x="BHK",
            y="Price",
            title="6. Overall Price Distribution by BHK",
            labels={"BHK": "BHK", "Price": "Price (INR)"},
            hover_data={"Price": ":,.0f"},
            color="BHK",
        )
        st.plotly_chart(fig_box_bhk_all, use_container_width=True)

        counts_df_city_all = df["city"].value_counts().reset_index()
        counts_df_city_all.columns = ["City_Category", "Count"]
        fig_count_city_all = px.bar(
            counts_df_city_all,
            x="City_Category",
            y="Count",
            title="7. Overall Property Count by City",
            labels={"City_Category": "City", "Count": "Count"},
            color="City_Category",
        )
        st.plotly_chart(fig_count_city_all, use_container_width=True)

        counts_df_balcony_all = df["Balcony"].value_counts().reset_index()
        counts_df_balcony_all.columns = ["Balcony_Status", "Count"]
        fig_count_balcony_all = px.bar(
            counts_df_balcony_all,
            x="Balcony_Status",
            y="Count",
            title="8. Overall Balcony Presence Count",
            labels={"Balcony_Status": "Balcony Available", "Count": "Count"},
            color="Balcony_Status",
        )
        st.plotly_chart(fig_count_balcony_all, use_container_width=True)


# --- AI-Powered Page ---
elif selected == "AI Search":
    st.title("🤖 AI-Integrated Smart Search")
    st.markdown("---")

    st.markdown(
        """
        <p style='font-size: 1.1em; text-align: center;'>
            Ask a natural language question about properties to find relevant listings.
        </p>
        <p style='font-size: 0.9em; text-align: center; color: #555;'>
            Example: "Show me 3BHK apartments in Bangalore under 1 Crore." <br>
            Example: "Find villas in Hyderabad with area more than 2000 sqft."
        </p>
        """,
        unsafe_allow_html=True,
    )

    user_query = st.text_area(
        "Enter your query:",
        placeholder="e.g., Find 2BHK flats in Pune with balcony",
        height=100,
    )

    if st.button("AI Search", key="ai_search_button"):
        if user_query:
            with st.spinner("Processing your query with AI..."):
                filtered_ai_df, extracted_filters = get_llm_filtered_data(user_query)

            st.markdown("---")
            st.subheader("AI Search Results")

            if extracted_filters:
                st.markdown("**Extracted Filters:**")
                # Display extracted filters in a nicely formatted bulleted list
                for key, value in extracted_filters.items():
                    # Format price in lakhs for display if key indicates price
                    if "price_lakhs" in key and value is not None:
                        st.markdown(
                            f"- **{key.replace('_', ' ').title()}:** ₹{value:.2f} Lakhs"
                        )
                    else:
                        st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
                st.markdown("---")

            if not filtered_ai_df.empty:
                st.info(f"Found {len(filtered_ai_df)} records matching your AI query.")

                # Create a display copy for AI results
                display_ai_df = filtered_ai_df.copy()

                # Remove 'Total_Rooms' column for display
                if "Total_Rooms" in display_ai_df.columns:
                    display_ai_df = display_ai_df.drop(columns=["Total_Rooms"])

                # Apply price formatting
                display_ai_df["Price"] = display_ai_df["Price"].apply(
                    format_price_for_display
                )

                st.dataframe(
                    display_ai_df.reset_index(drop=True).rename(index=lambda x: x + 1)
                )

                # Optional: Basic visualizations for AI-filtered data
                if (
                    len(filtered_ai_df) > 5
                ):  # Only show plots if a reasonable number of results
                    st.markdown("### Visualizations for AI Search Results")

                    plot_ai_col1, plot_ai_col2 = st.columns(2)

                    with plot_ai_col1:
                        fig_ai_price = px.histogram(
                            filtered_ai_df,
                            x="Price",
                            title="Price Distribution (AI Filtered)",
                            labels={"Price": "Price (INR)"},
                            hover_data={"Price": ":,.0f"},
                        )
                        st.plotly_chart(fig_ai_price, use_container_width=True)

                        if len(filtered_ai_df["property_type"].unique()) > 1:
                            counts_df_ai_prop = (
                                filtered_ai_df["property_type"]
                                .value_counts()
                                .reset_index()
                            )
                            counts_df_ai_prop.columns = [
                                "Property_Type_Category",
                                "Count",
                            ]
                            fig_ai_prop_count = px.bar(
                                counts_df_ai_prop,
                                x="Property_Type_Category",
                                y="Count",
                                title="Property Count by Type (AI Filtered)",
                                labels={
                                    "Property_Type_Category": "Property Type",
                                    "Count": "Count",
                                },
                                color="Property_Type_Category",
                            )
                            st.plotly_chart(fig_ai_prop_count, use_container_width=True)

                    with plot_ai_col2:
                        fig_ai_area_price = px.scatter(
                            filtered_ai_df,
                            x="Total_Area(SQFT)",
                            y="Price",
                            color="BHK",
                            title="Area vs Price (AI Filtered)",
                            labels={
                                "Total_Area(SQFT)": "Total Area (SQFT)",
                                "Price": "Price (INR)",
                            },
                            hover_data={
                                "Price": ":,.0f",
                                "Total_Area(SQFT)": True,
                                "BHK": True,
                            },
                        )
                        st.plotly_chart(fig_ai_area_price, use_container_width=True)

                        if len(filtered_ai_df["city"].unique()) > 1:
                            counts_df_ai_city = (
                                filtered_ai_df["city"].value_counts().reset_index()
                            )
                            counts_df_ai_city.columns = ["City_Category", "Count"]
                            fig_ai_city_count = px.bar(
                                counts_df_ai_city,
                                x="City_Category",
                                y="Count",
                                title="Property Count by City (AI Filtered)",
                                labels={"City_Category": "City", "Count": "Count"},
                                color="City_Category",
                            )
                            st.plotly_chart(fig_ai_city_count, use_container_width=True)

            else:
                st.warning(
                    "No properties found matching your AI query. Try rephrasing or broadening your search."
                )
        else:
            st.warning("Please enter a query to perform an AI search.")
