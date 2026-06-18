from time import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google import genai
import os
import json
import joblib

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

client = genai.Client(
    api_key="Gen_Api_Key"
)

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SaaS Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)  # <--- Fixed!
# -------------------------------------------------------------------
# 2. CACHED DATA INGESTION & FEATURE ENGINEERING
# -------------------------------------------------------------------
# -----------------------------
# Generate Business Report
# -----------------------------
# @st.cache_data
# def generate_report(summary):
#     prompt = f"""
#     You are a Chief Business Intelligence Officer.

#     Analyze the following Customer Intelligence Summary:

#     {json.dumps(summary, indent=2)}

#     Provide:

#     1. Executive Summary
#     2. Customer Churn Analysis
#     3. Revenue Forecast Analysis
#     4. Cluster Performance Analysis
#     5. Retention Strategy Recommendations
#     6. Revenue Growth Opportunities
#     7. Business Risks
#     8. Action Plan for Management

#     Provide professional business insights.
#     """

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=prompt
#     )

#     return response.text
@st.cache_data
def generate_report(summary):

    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    prompt = f"""
    You are a Chief Business Intelligence Officer.

    Analyze the following Customer Intelligence Summary:

    {json.dumps(summary, indent=2, default=convert_numpy)}

    Provide:

    1. Executive Summary
    2. Customer Churn Analysis
    3. Revenue Forecast Analysis
    4. Cluster Performance Analysis
    5. Retention Strategy Recommendations
    6. Revenue Growth Opportunities
    7. Business Risks
    8. Action Plan for Management

    Provide professional business insights.
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text

        except Exception as e:

            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return f"""
# AI Report Generation Failed

Reason:
{str(e)}

The Gemini API is currently unavailable due to high demand.

Please try again after a few minutes.
"""

# -------------------------------------------------------------------
# 4. MAIN EXECUTIVE DASHBOARD LAYOUT
# -------------------------------------------------------------------
st.title("💼 Enterprise AI Customer Analytics Framework")
st.markdown("Strategic Operations Center for Subscription Health & Revenue Forecasts")
st.markdown("---")


# -------------------------------------------------------------------
# 5. CORE ANALYTICAL GRAPHICS TABS
# -------------------------------------------------------------------
# Load Model
@st.cache_resource
def load_churn_model():
    return joblib.load("churn_prediction_model.pkl")

@st.cache_resource
def load_revenue_model():
    return joblib.load("revenue_prediction_model.pkl")

@st.cache_resource
def load_segmentation_models():
    preprocess = joblib.load("output/segmentation_preprocess.pkl")
    kmeans = joblib.load("output/segmentation_kmeans.pkl")
    return preprocess, kmeans

tab1, tab2, tab3, tab4 = st.tabs([
    "Customer Churn Prediction System",
    "Revenue Forecasting Dashboard",
    "Intelligent Customer Segmentation",
    "AI Business Insights"
])

with tab1:
    st.subheader("Account Strategic Matrix Position")
    st.markdown("Visualizing accounts based on Churn Risk vs. Revenue Expansion Score. Focus on top right for growth, top left for critical retention saves.")
    st.title("Customer Churn Prediction System")



    model = load_churn_model()

    # Upload CSV
    uploaded_file = st.file_uploader(
        "Upload Churn Customer CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Customer ID Column
        customer_id_col = "Customer_ID"  # Change according to your dataset

        if customer_id_col not in df.columns:
            st.error(f"'{customer_id_col}' column not found!")
        else:
            # Save Customer IDs
            customer_ids = df[customer_id_col]
            # Remove ID column before prediction
            X = df

            try:
                # Prediction
                predictions = model.predict(X)
                churn_probabilities = model.predict_proba(X)[:, 1]

                # Add Prediction Column
                result_df = df.copy()
                # st.write(result_df.columns.tolist())
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                with kpi1:
                    with st.container(border=True):
                        st.metric("👥 Total Customers", f"{len(result_df):,}")

                with kpi2:
                    with st.container(border=True):
                        st.metric(
                            "💰 Avg Revenue",
                            f"${result_df['avg_monthly_revenue'].mean():,.0f}"
                        )

                with kpi3:
                    with st.container(border=True):
                        st.metric(
                            "❤️ Avg Health Score",
                            f"{result_df['Health_Score'].mean():.1f}"
                        )

                with kpi4:
                    high_value = (result_df['Health_Score'] >= 80).sum()

                    with st.container(border=True):
                        st.metric(
                            "⭐ High Health Customers",
                            f"{high_value:,}"
                        )


                result_df["Churn_Prediction"] = predictions
                result_df["Churn_Probability"] = (
                        churn_probabilities * 100
                    ).round(2)
                # Filter Churn Customers
                churn_customers = result_df[
                    result_df["Churn_Prediction"] == 1
                ][[customer_id_col, "Churn_Prediction","Churn_Probability"]]

                st.subheader("Predicted Churn Customers")

                if len(churn_customers) > 0:
                    st.dataframe(churn_customers)

                    csv = churn_customers.to_csv(index=False)

                    st.download_button(
                        label="Download Churn Customers",
                        data=csv,
                        file_name="churn_customers.csv",
                        mime="text/csv"
                    )

                    st.success(
                        f"Total Churn Customers: {len(churn_customers)}"
                    )

                else:
                    st.success("No churn customers predicted.")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
            if not result_df.empty:
                # Prevent auto ID collision with unique key assignment
                fig = fig = px.scatter(
                    result_df,
                    x="Health_Score",
                    y="Renewal_Probability",
                    color="Churn_Prediction",
                    hover_name="Customer_ID"
                )

                st.plotly_chart(fig, use_container_width=True)
                fig.update_layout(paper_bgcolor='white', plot_bgcolor='rgba(248,249,250,1)', height=600)
                # st.plotly_chart(fig, use_container_width=True, key="strategic_matrix_scatter")
            else:
                st.info("No records found matching current criteria matrix filters.")

with tab2:
    st.set_page_config(page_title="Revenue Forecasting", layout="wide")

    st.title("Revenue Forecasting Dashboard")

    # Load Model
    model = load_revenue_model()

    uploaded_file = st.file_uploader(
        "Upload Revenue CSV",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        customer_id_col = "Customer_ID"

        X = df
        revenue_predictions = model.predict(X)

        total_revenue = revenue_predictions.sum()

        # st.success(
        #     f"Current Predicted Revenue: ₹{total_revenue:,.2f}"
        # )

        # User Inputs
        periods = st.number_input(
            "Forecast Period",
            min_value=1,
            value=5
        )

        forecast_type = st.selectbox(
            "Forecast Type",
            ["Days", "Weeks", "Months", "Quarters", "Years"]
        )

        # Growth Assumptions
        growth_rates = {
            "Days": 0.005,
            "Weeks": 0.02,
            "Months": 0.05,
            "Quarters": 0.10,
            "Years": 0.20
        }

        growth = growth_rates[forecast_type]

        forecast_data = []

        current = total_revenue

        for i in range(1, periods + 1):

            current = current * (1 + growth)

            forecast_data.append({
                forecast_type: i,
                "Forecast Revenue": current
            })

        forecast_df = pd.DataFrame(forecast_data)

        st.subheader(
            f"Next {periods} {forecast_type} Revenue Forecast"
        )

        st.dataframe(forecast_df)

        # Plot
        fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(
            forecast_df[forecast_type],
            forecast_df["Forecast Revenue"],
            marker="o"
        )

        ax.set_title(
            f"Revenue Forecast for Next {periods} {forecast_type}"
        )

        ax.set_xlabel(forecast_type)
        ax.set_ylabel("Revenue")

        st.pyplot(fig)

        # Download
        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "Download Forecast",
            csv,
            "revenue_forecast.csv",
            "text/csv"
        )
       
        

with tab3:
    # ------------------------------------
    # Load Saved Models
    # ------------------------------------
    preprocess, kmeans = load_segmentation_models()

    # ------------------------------------
    # Cluster Mapping
    # ------------------------------------
    cluster_map = {
        0: "Mid-Tier Active Customers",
        1: "High-Revenue Enterprise Customers",
        2: "High-Value Growth Customers",
        3: "At-Risk Low Engagement Customers"
    }

    st.set_page_config(
        page_title="Customer Segmentation",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Intelligent Customer Segmentation")

    st.write(
        "Upload a customer CSV file and get customer segment predictions."
    )

    # ------------------------------------
    # Upload CSV
    # ------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        try:

            # Save Customer_ID separately
            customer_ids = df["Customer_ID"]
            result_df = df.copy()
            # Remove columns used during training
            drop_cols = [
                'Customer_ID',
                'Geo_ID',
                'Industry_ID',
                'Product_ID'
            ]

            X = df.drop(
                columns=[c for c in drop_cols if c in df.columns],
                errors="ignore"
            )

            # ------------------------------------
            # Transform Data
            # ------------------------------------
            X_processed = preprocess.transform(X)

            # ------------------------------------
            # Predict Cluster
            # ------------------------------------
            clusters = kmeans.predict(X_processed)

            # ------------------------------------
            # Result DataFrame
            # ------------------------------------
            result_df = pd.DataFrame({
                "Customer_ID": customer_ids,
                "Cluster": clusters
            })

            result_df["Cluster_Name"] = result_df["Cluster"].map(cluster_map)

            st.subheader("Segmentation Results")

            st.dataframe(result_df)

            # ------------------------------------
            # Cluster Distribution
            # ------------------------------------
            st.subheader("Cluster Distribution")

            chart_df = result_df["Cluster_Name"].value_counts()

            st.bar_chart(chart_df)

            # ------------------------------------
            # Download Results
            # ------------------------------------
            csv = result_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="customer_segmentation_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
with tab4:
    # col1,col2=st.columns(2)
    # with col1:
    #     st.subheader("📈 Churn Risk Distribution")

    #     fig = px.histogram(
    #         filtered_df,
    #         x="churn_risk_score",
    #         nbins=20,
    #         color="action_segment",
    #         color_discrete_map={
    #             "PROTECT":"#dc3545",
    #             "GROW":"#28a745",
    #             "NURTURE":"#ffc107",
    #             "MONITOR":"#17a2b8"
    #         }
    #     )

    #     st.plotly_chart(fig, use_container_width=True)  
    # with col2:
    #     st.subheader("🚨 Average Churn Risk by Cluster")

    #     cluster_risk = (
    #         filtered_df
    #         .groupby("Cluster_Name")["churn_risk_score"]
    #         .mean()
    #         .reset_index()
    #     )

    #     fig = px.bar(
    #         cluster_risk,
    #         x="Cluster_Name",
    #         y="churn_risk_score",
    #         color="churn_risk_score",
    #         color_continuous_scale="Reds"
    #     )

    #     st.plotly_chart(fig, use_container_width=True)
    
    
    # col1,col2=st.columns(2)
    # with col1:
    #     st.subheader("💰 Revenue vs Churn Risk")

    #     fig = px.scatter(
    #         filtered_df,
    #         x="Predicted_Revenue",
    #         y="churn_risk_score",
    #         color="Cluster_Name",
    #         size="Predicted_Revenue",
    #         hover_name="Customer_ID"
    #     )

    #     st.plotly_chart(fig, use_container_width=True)
    # with col2:
    #     st.subheader("🏆 Revenue Contribution by Cluster")

    #     cluster_rev = (
    #         filtered_df
    #         .groupby("Cluster_Name")["Predicted_Revenue"]
    #         .sum()
    #         .reset_index()
    #     )

    #     fig = px.pie(
    #         cluster_rev,
    #         names="Cluster_Name",
    #         values="Predicted_Revenue",
    #         hole=0.5
    #     )

    #     st.plotly_chart(fig, use_container_width=True)
    
    # col1,col2=st.columns(2)
    # with col1:
    #     st.subheader("🔥 Top 20 Churn-Risk Customers")

    #     top20 = filtered_df.nlargest(
    #         20,
    #         "churn_risk_score"
    #     )[[
    #         "Customer_ID",
    #         "Cluster_Name",
    #         "churn_risk_score",
    #         "Predicted_Revenue"
    #     ]]

    #     st.dataframe(top20, use_container_width=True)
    # with col2:
    #     st.subheader("🎯 Strategic Portfolio Matrix")

    #     fig = px.scatter(
    #         filtered_df,
    #         x="churn_risk_score",
    #         y="revenue_opportunity_score",
    #         color="action_segment",
    #         size="Predicted_Revenue",
    #         hover_name="Customer_ID"
    #     )

    #     fig.add_vline(x=50, line_dash="dash", line_color="red")
    #     fig.add_hline(y=40, line_dash="dash", line_color="green")

    #     st.plotly_chart(fig, use_container_width=True)

    # st.subheader("🤖 Gemini Strategic Intelligence")

    # if st.button("Generate AI Insights"):

    #     with st.spinner("Analyzing customer portfolio..."):

    #         try:
    #             insights = get_ai_insights(filtered_df)

    #             st.success("Analysis Complete")

    #             st.markdown(insights)

    #         except Exception as e:
    #             st.error(f"Gemini Error: {e}")


    st.title("📊 Customer Intelligence Dashboard")

    # ---------------------------------------------------
    # Upload Files
    # ---------------------------------------------------
    churn_file = st.file_uploader(
    "Upload Customer Churn CSV File",
    type=["csv"],
    key="churn_upload"
)

    revenue_file = st.file_uploader(
        "Upload Revenue CSV File",
        type=["csv"],
        key="revenue_upload"
    )

    segment_file = st.file_uploader(
        "Upload Segmentation CSV File",
        type=["csv"],
        key="segment_upload"
    )

    # ---------------------------------------------------
    # Process Files
    # ---------------------------------------------------
    if churn_file and revenue_file and segment_file:

        # Read Files
        churn_df = pd.read_csv(churn_file)
        revenue_df = pd.read_csv(revenue_file)
        segment_df = pd.read_csv(segment_file)

        # st.write("Churn rows:", len(result_df))
        # st.write("Revenue rows:", len(revenue_df))
        # st.write("Segment rows:", len(segment_df))

        # =====================================================
        # CHURN PREDICTION
        # =====================================================

        churn_model = load_churn_model()

        churn_X = churn_df.copy()

        predictions = churn_model.predict(churn_X)

        churn_probabilities = (
            churn_model.predict_proba(churn_X)[:, 1]
        )

        result_df = churn_df.copy()

        result_df["Churn_Prediction"] = predictions

        result_df["Churn_Probability"] = (
            churn_probabilities * 100
        ).round(2)

        # =====================================================
        # REVENUE PREDICTION
        # =====================================================

        revenue_model = load_revenue_model()

        revenue_X = revenue_df.copy()

        revenue_predictions = revenue_model.predict(
            revenue_X
        )

        revenue_df["Predicted_Revenue"] = revenue_predictions

        total_revenue = revenue_predictions.sum()

        st.success(
            f"Current Predicted Revenue: ₹{total_revenue:,.2f}"
        )

        # =====================================================
        # REVENUE FORECAST
        # =====================================================

        periods = st.number_input(
            "Forecast Period",
            min_value=1,
            value=5,
            step=1,
            key="customer_dashboard_forecast_period"
        )

        forecast_type = st.selectbox(
            "Forecast Type",
            [
                "Days",
                "Weeks",
                "Months",
                "Quarters",
                "Years"
            ],
            key="customer_dashboard_forecast_type"
        )

        growth_rates = {
            "Days": 0.005,
            "Weeks": 0.02,
            "Months": 0.05,
            "Quarters": 0.10,
            "Years": 0.20
        }

        growth = growth_rates[forecast_type]

        forecast_data = []

        current = total_revenue

        for i in range(1, periods + 1):

            current = current * (1 + growth)

            forecast_data.append({
                forecast_type: i,
                "Forecast_Revenue": round(current, 2)
            })

        forecast_df = pd.DataFrame(
            forecast_data
        )

        st.subheader(
            f"Revenue Forecast ({periods} {forecast_type})"
        )

        st.dataframe(forecast_df)

        # =====================================================
        # CUSTOMER SEGMENTATION
        # =====================================================

        customer_ids = segment_df["Customer_ID"]

        drop_cols = [
            "Customer_ID",
            "Geo_ID",
            "Industry_ID",
            "Product_ID"
        ]

        X_segment = segment_df.drop(
            columns=[
                c for c in drop_cols
                if c in segment_df.columns
            ],
            errors="ignore"
        )

        X_processed = preprocess.transform(
            X_segment
        )

        clusters = kmeans.predict(
            X_processed
        )

        segment_result = pd.DataFrame({
            "Customer_ID": customer_ids,
            "Cluster": clusters
        })

        segment_result["Cluster_Name"] = (
            segment_result["Cluster"].map(
                cluster_map
            )
        )
        # st.write("Churn rows:", len(result_df))
        # st.write("Revenue rows:", len(revenue_df))
        # st.dataframe(result_df)
        # st.write("Segment rows:", len(segment_df))
        # =====================================================
        # CUSTOMER INTELLIGENCE PROFILE
        # =====================================================

        # customer_profile = (
        #     result_df
        #     .merge(
        #         revenue_df[
        #             [
        #                 "Customer_ID",
        #                 "Predicted_Revenue"
        #             ]
        #         ],
        #         on="Customer_ID",
        #         how="inner"
        #     )
        #     .merge(
        #         segment_result,
        #         on="Customer_ID",
        #         how="inner"
        #     )
        # )
        customer_profile = (
    result_df
    .merge(
        revenue_df[["Customer_ID", "Predicted_Revenue"]],
        on="Customer_ID",
        how="inner"
    )
    .merge(
        segment_result,
        on="Customer_ID",
        how="inner"
    )
)
        st.success(
            "Files merged successfully!"
        )

        st.subheader(
            "Customer Intelligence Profile"
        )

        st.dataframe(
            customer_profile.head()
        )

        # =====================================================
        # RISK LEVEL
        # =====================================================

        customer_profile["Risk_Level"] = (
            customer_profile[
                "Churn_Probability"
            ].apply(
                lambda x:
                "High Risk"
                if x >= 70
                else "Low Risk"
            )
        )

        # =====================================================
        # REVENUE CATEGORY
        # =====================================================

        customer_profile[
            "Revenue_Category"
        ] = customer_profile[
            "Predicted_Revenue"
        ].apply(
            lambda x:
            "High Revenue"
            if x >= 5000
            else "Low Revenue"
        )

        # =====================================================
        # BUSINESS SUMMARY
        # =====================================================

        summary1 = {

            "Total_Customers":
                int(len(customer_profile)),

            "Average_Churn_Probability":
                round(
                    customer_profile[
                        "Churn_Probability"
                    ].mean(),
                    2
                ),

            "High_Risk_Customers":
                int(
                    len(
                        customer_profile[
                            customer_profile[
                                "Risk_Level"
                            ] == "High Risk"
                        ]
                    )
                ),

            "Low_Risk_Customers":
                int(
                    len(
                        customer_profile[
                            customer_profile[
                                "Risk_Level"
                            ] == "Low Risk"
                        ]
                    )
                ),

            "Total_Predicted_Revenue":
                round(
                    customer_profile[
                        "Predicted_Revenue"
                    ].sum(),
                    2
                ),

            "Average_Predicted_Revenue":
                round(
                    customer_profile[
                        "Predicted_Revenue"
                    ].mean(),
                    2
                ),

            "Cluster_Distribution":
                customer_profile[
                    "Cluster_Name"
                ].value_counts().to_dict(),

            "Revenue_Distribution":
                customer_profile[
                    "Revenue_Category"
                ].value_counts().to_dict(),

            "Risk_Distribution":
                customer_profile[
                    "Risk_Level"
                ].value_counts().to_dict()
        }
       
        # =====================================================
        # EXECUTIVE REPORT
        # =====================================================

        if st.button(
            "Generate Executive Report"
        ):

            with st.spinner(
                "Generating Business Insights..."
            ):

                report = generate_report(
                    summary1
                )

            tab1,tab5 = st.tabs([
                "Executive Summary",
                "Full Report"
            ])

            with tab1:
                st.subheader(
                    "Executive Summary"
                )
                st.markdown(report)            

            with tab5:

                st.subheader(
                    "Complete Report"
                )

                st.text_area(
                    "Generated Report",
                    report,
                    height=600
                )

                st.download_button(
                    "Download Report",
                    report,
                    file_name="business_report.txt",
                    mime="text/plain"
                )

                customer_profile_csv = (
                    customer_profile.to_csv(
                        index=False
                    )
                )

                st.download_button(
                    "Download Customer Profile",
                    customer_profile_csv,
                    file_name="customer_intelligence_profile.csv",
                    mime="text/csv"
                )
