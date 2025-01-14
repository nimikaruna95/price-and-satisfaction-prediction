import streamlit as st
import os

# page title
st.set_page_config(page_title="Flight Customer App", layout="wide")

# App title
st.title("Flight Customer App")

# Navigate between pages
page = st.sidebar.radio("Select a page", ["Customer Satisfaction Prediction", "Flight Price Prediction"])

# Customer Satisfaction Prediction Page
if page == "Customer Satisfaction Prediction":
    import customer_satisfaction.st2 as st2
    st2.main()

# Flight Price Prediction Page
elif page == "Flight Price Prediction":
    import flight_price.st1 as st1
    st1.main()


