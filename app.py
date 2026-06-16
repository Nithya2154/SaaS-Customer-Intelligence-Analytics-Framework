import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google import genai
import os
import json

client = genai.Client(
    api_key="Gen_Api_Key_Here"  # safer than hardcoding
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
@st.cache_data(ttl=3600)
def get_ai_insights(df):

    client = genai.Client(
        api_key="Gen_Api_Key_Here"  # safer than hardcoding
    )

    sample_data = df[[
        "Customer_ID",
        "Churn_Probability",
        "Predicted_Revenue",
        "Cluster_Name"
    ]].head(100)

    prompt = f"""
    Analyze the SaaS customer dataset.

    Dataset Sample:
    {sample_data.to_csv(index=False)}

    Answer:

    1. What churn trends do you observe?
    2. How does revenue relate to clusters?
    3. What customer segmentation strategy do you recommend?
    4. What data quality concerns do you notice?

    Give concise business recommendations.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
# -----------------------------
# Generate Business Report
# -----------------------------
@st.cache_data
def generate_report(summary):
    prompt = f"""
    You are a Chief Business Intelligence Officer.

    Analyze the following Customer Intelligence Summary:

    {json.dumps(summary, indent=2)}

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

@st.cache_data
def load_and_process_data():
    # Load your unified intelligence matrix
    df = pd.read_csv("customer_intelligence_profile.csv")
    
    # Drop exact duplicates if any exist from outer merges
    df = df.drop_duplicates(subset=['Customer_ID']).copy()
    
    # Mocking standard business baseline metrics if missing from dataset for visuals
    if 'Current_ACV' not in df.columns:
        np.random.seed(42)
        df['Current_ACV'] = df['Predicted_Revenue'] * np.random.uniform(0.75, 0.95, len(df))
    
    # Recalculate Risk & Opportunity Framework Metrics
    df['churn_risk_score'] = (df['Churn_Probability'] * 100).round(1)
    
    # Opportunity value is defined by the upward predictive variance
    df['revenue_variance'] = df['Predicted_Revenue'] - df['Current_ACV']
    max_var = df['revenue_variance'].max() if df['revenue_variance'].max() > 0 else 1
    df['revenue_opportunity_score'] = ((df['revenue_variance'].clip(lower=0) / max_var) * 100).round(1)
    
    # Segment assignment boundaries mapping risk vs upside
    def assign_action_segment(row):
        if row['churn_risk_score'] >= 50:
            return "PROTECT"
        elif row['revenue_opportunity_score'] >= 40:
            return "GROW"
        elif row['churn_risk_score'] < 20 and row['revenue_opportunity_score'] < 20:
            return "MONITOR"
        else:
            return "NURTURE"
            
    df['action_segment'] = df.apply(assign_action_segment, axis=1)
    return df

try:
    df_master = load_and_process_data()
except Exception as e:
    st.error(f"Error reading unified dataset profile: {e}")
    st.stop()

# -------------------------------------------------------------------
# 3. GLOBAL SIDEBAR FILTER CONTROLS
# -------------------------------------------------------------------
st.sidebar.title("🎛️ Strategic Filters")
st.sidebar.markdown("---")

# Segment selection filter
available_segments = sorted(df_master['action_segment'].unique())
selected_segments = st.sidebar.multiselect(
    "Priority Strategy Segments",
    options=available_segments,
    default=available_segments
)

# Cluster cohort filter
available_clusters = sorted(df_master['Cluster_Name'].dropna().unique())
selected_clusters = st.sidebar.multiselect(
    "AI Customer Cohort Clusters",
    options=available_clusters,
    default=available_clusters
)

# Risk Threshold Slider
risk_threshold = st.sidebar.slider(
    "Minimum Churn Probability Filter (%)",
    min_value=0.0, max_value=100.0, value=0.0, step=5.0
)

# Filter Dataset Downward Stream
filtered_df = df_master[
    (df_master['action_segment'].isin(selected_segments)) &
    (df_master['Cluster_Name'].isin(selected_clusters)) &
    (df_master['churn_risk_score'] >= risk_threshold)
]

# -------------------------------------------------------------------
# 4. MAIN EXECUTIVE DASHBOARD LAYOUT
# -------------------------------------------------------------------
st.title("💼 Enterprise AI Customer Analytics Framework")
st.markdown("Strategic Operations Center for Subscription Health & Revenue Forecasts")
st.markdown("---")

# High-Level KPIs Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-box">
        <h6 style="color:#6c757d; margin:0;">ACTIVE PORTFOLIO SIZE</h6>
        <h2 style="margin:5px 0;">{len(filtered_df):,}</h2>
        <small style="color:#28a745;">Filtered Accounts</small>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    at_risk_count = len(filtered_df[filtered_df['action_segment'] == "PROTECT"])
    st.markdown(f"""
    <div class="metric-box" style="border-left-color: #dc3545;">
        <h6 style="color:#6c757d; margin:0;">ACCOUNTS AT RISK</h6>
        <h2 style="margin:5px 0; color:#dc3545;">{at_risk_count:,}</h2>
        <small style="color:#6c757d;">Segment: PROTECT</small>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    total_current_acv = filtered_df['Current_ACV'].sum()
    st.markdown(f"""
    <div class="metric-box">
        <h6 style="color:#6c757d; margin:0;">TOTAL RUN-RATE ACV</h6>
        <h2 style="margin:5px 0;">${total_current_acv:,.0f}</h2>
        <small style="color:#6c757d;">Current Baseline Portfolio</small>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    total_pred_rev = filtered_df['Predicted_Revenue'].sum()
    rev_delta = total_pred_rev - total_current_acv
    delta_color = "#28a745" if rev_delta >= 0 else "#dc3545"
    st.markdown(f"""
    <div class="metric-box" style="border-left-color: #28a745;">
        <h6 style="color:#6c757d; margin:0;">AI PREDICTED REVENUE</h6>
        <h2 style="margin:5px 0; color:#28a745;">${total_pred_rev:,.0f}</h2>
        <small style="color:{delta_color}; font-weight:bold;">
            {'+$' if rev_delta >=0 else '-$'}{abs(rev_delta):,.0f} Upside
        </small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 5. CORE ANALYTICAL GRAPHICS TABS
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk vs Opportunity Mapping",
    "📊 Segment Breakdown",
    "🔍 Granular Account Explorer",
    "🤖 AI Business Insights"
])

with tab1:
    st.subheader("Account Strategic Matrix Position")
    st.markdown("Visualizing accounts based on Churn Risk vs. Revenue Expansion Score. Focus on top right for growth, top left for critical retention saves.")
    
    if not filtered_df.empty:
        # Prevent auto ID collision with unique key assignment
        fig = px.scatter(
            filtered_df,
            x="churn_risk_score",
            y="revenue_opportunity_score",
            color="action_segment",
            size="Predicted_Revenue",
            hover_name="Customer_ID",
            hover_data=["Cluster_Name", "Current_ACV", "Predicted_Revenue"],
            labels={
                "churn_risk_score": "Churn Risk Score (%)",
                "revenue_opportunity_score": "Revenue Opportunity Score (%)",
                "action_segment": "Strategic Assignment"
            },
            color_discrete_map={"PROTECT": "#dc3545", "GROW": "#28a745", "NURTURE": "#ffc107", "MONITOR": "#17a2b8"}
        )
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='rgba(248,249,250,1)', height=600)
        st.plotly_chart(fig, use_container_width=True, key="strategic_matrix_scatter")
    else:
        st.info("No records found matching current criteria matrix filters.")

with tab2:
    left_chart, right_chart = st.columns(2)
    
    # Custom static map colors matching professional dashboard architecture
    color_map = {"PROTECT": "#dc3545", "GROW": "#28a745", "NURTURE": "#ffc107", "MONITOR": "#17a2b8"}
    
    with left_chart:
        st.markdown("#### Distribution of Customers Across Strategic Quadrants")
        segment_counts = filtered_df['action_segment'].value_counts().reset_index()
        if not segment_counts.empty:
            fig_pie = px.pie(
                segment_counts, 
                values='count', 
                names='action_segment',
                color='action_segment',
                color_discrete_map=color_map,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="quadrant_distribution_pie")
        else:
            st.info("No segments available.")

    with right_chart:
        st.markdown("#### Baseline vs AI Predicted Revenue Outlooks by Clusters")
        cluster_rev = filtered_df.groupby('Cluster_Name')[['Current_ACV', 'Predicted_Revenue']].sum().reset_index()
        
        if not cluster_rev.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Current ACV Run Rate',
                x=cluster_rev['Cluster_Name'], y=cluster_rev['Current_ACV'],
                marker_color='#4A90E2'
            ))
            fig_bar.add_trace(go.Bar(
                name='AI Predicted Forecast',
                x=cluster_rev['Cluster_Name'], y=cluster_rev['Predicted_Revenue'],
                marker_color='#2BA14B'
            ))
            fig_bar.update_layout(barmode='group', height=400, xaxis_tickangle=-15)
            st.plotly_chart(fig_bar, use_container_width=True, key="revenue_outlook_bar")
        else:
            st.info("No data available to display cluster comparison forecasts.")

with tab3:
    st.subheader("Prioritized Actions Data Ledger")
    st.markdown("Sort and search down the complete customer intelligence records dataset to triage daily workflows.")
    
    # Clean visual representations for tables
    display_cols = [
        'Customer_ID', 'Cluster_Name', 'churn_risk_score', 
        'revenue_opportunity_score', 'Current_ACV', 'Predicted_Revenue', 'action_segment'
    ]
    
    styled_df = filtered_df[display_cols].copy()
    styled_df.columns = [
        'Customer ID', 'AI Cluster Designation', 'Churn Risk (%)', 
        'Upside Score', 'Current ACV', 'Forecast Revenue', 'Action Plan'
    ]
    
    st.dataframe(
        styled_df.sort_values(by=['Churn Risk (%)', 'Upside Score'], ascending=[False, False]),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    col1,col2=st.columns(2)
    with col1:
        st.subheader("📈 Churn Risk Distribution")

        fig = px.histogram(
            filtered_df,
            x="churn_risk_score",
            nbins=20,
            color="action_segment",
            color_discrete_map={
                "PROTECT":"#dc3545",
                "GROW":"#28a745",
                "NURTURE":"#ffc107",
                "MONITOR":"#17a2b8"
            }
        )

        st.plotly_chart(fig, use_container_width=True)  
    with col2:
        st.subheader("🚨 Average Churn Risk by Cluster")

        cluster_risk = (
            filtered_df
            .groupby("Cluster_Name")["churn_risk_score"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            cluster_risk,
            x="Cluster_Name",
            y="churn_risk_score",
            color="churn_risk_score",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(fig, use_container_width=True)
    
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader("💰 Revenue vs Churn Risk")

        fig = px.scatter(
            filtered_df,
            x="Predicted_Revenue",
            y="churn_risk_score",
            color="Cluster_Name",
            size="Predicted_Revenue",
            hover_name="Customer_ID"
        )

        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🏆 Revenue Contribution by Cluster")

        cluster_rev = (
            filtered_df
            .groupby("Cluster_Name")["Predicted_Revenue"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            cluster_rev,
            names="Cluster_Name",
            values="Predicted_Revenue",
            hole=0.5
        )

        st.plotly_chart(fig, use_container_width=True)
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader("🔥 Top 20 Churn-Risk Customers")

        top20 = filtered_df.nlargest(
            20,
            "churn_risk_score"
        )[[
            "Customer_ID",
            "Cluster_Name",
            "churn_risk_score",
            "Predicted_Revenue"
        ]]

        st.dataframe(top20, use_container_width=True)
    with col2:
        st.subheader("🎯 Strategic Portfolio Matrix")

        fig = px.scatter(
            filtered_df,
            x="churn_risk_score",
            y="revenue_opportunity_score",
            color="action_segment",
            size="Predicted_Revenue",
            hover_name="Customer_ID"
        )

        fig.add_vline(x=50, line_dash="dash", line_color="red")
        fig.add_hline(y=40, line_dash="dash", line_color="green")

        st.plotly_chart(fig, use_container_width=True)

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

    if st.button("Generate Executive Report"):

        with st.spinner("Generating Business Insights..."):
            with open("business_summary.json", "r") as f:
                summary = json.load(f)
            report = generate_report(summary)

        # Create 5 Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Executive Summary",
            "Churn Analysis",
            "Revenue Analysis",
            "Recommendations",
            "Full Report"
        ])

        with tab1:
            st.subheader("Executive Summary")
            st.markdown(report)

        with tab2:
            st.subheader("Customer Churn Analysis")
            st.markdown(report)

        with tab3:
            st.subheader("Revenue Forecast Analysis")
            st.markdown(report)

        with tab4:
            st.subheader("Business Recommendations")
            st.markdown(report)

        with tab5:
            st.subheader("Complete Business Intelligence Report")
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