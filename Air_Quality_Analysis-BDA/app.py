
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import silhouette_score

# Page settings
st.set_page_config(
    page_title="AQI Analysis Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Load project files
folder = os.path.dirname(os.path.abspath(__file__))

city = pd.read_csv(os.path.join(folder, "city_summary.csv"))
year = pd.read_csv(os.path.join(folder, "city_year.csv"))
month = pd.read_csv(os.path.join(folder, "monthly.csv"))
bucket = pd.read_csv(os.path.join(folder, "bucket_counts.csv"))

model = joblib.load(os.path.join(folder, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
features = joblib.load(os.path.join(folder, "features.pkl"))

# AQI category function
def aqi_category(v):
    if pd.isna(v):
        return "Unknown"
    elif v <= 50:
        return "Good"
    elif v <= 100:
        return "Satisfactory"
    elif v <= 200:
        return "Moderate"
    elif v <= 300:
        return "Poor"
    elif v <= 400:
        return "Very Poor"
    return "Severe"


advice = {
    "Good": "Air quality is good. Enjoy outdoor activities.",
    "Satisfactory": "Air quality is acceptable for most people.",
    "Moderate": "Sensitive individuals should consider reducing prolonged outdoor exertion.",
    "Poor": "Consider reducing prolonged outdoor exertion.",
    "Very Poor": "Reduce outdoor exertion and follow local health guidance.",
    "Severe": "Avoid unnecessary outdoor exertion and follow local health guidance.",
    "Unknown": "AQI category is unavailable."
}

coords = {
    "Ahmedabad": (23.02, 72.57),
    "Aizawl": (23.73, 92.72),
    "Amaravati": (16.51, 80.52),
    "Amritsar": (31.63, 74.87),
    "Bengaluru": (12.97, 77.59),
    "Bhopal": (23.26, 77.41),
    "Brajrajnagar": (21.82, 83.92),
    "Chandigarh": (30.73, 76.78),
    "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96),
    "Delhi": (28.61, 77.21),
    "Ernakulam": (9.98, 76.28),
    "Gurugram": (28.46, 77.03),
    "Guwahati": (26.14, 91.74),
    "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79),
    "Jorapokhar": (23.71, 86.41),
    "Kochi": (9.93, 76.27),
    "Kolkata": (22.57, 88.36),
    "Lucknow": (26.85, 80.95),
    "Mumbai": (19.08, 72.88),
    "Patna": (25.59, 85.14),
    "Shillong": (25.57, 91.88),
    "Talcher": (20.95, 85.22),
    "Thiruvananthapuram": (8.52, 76.94),
    "Visakhapatnam": (17.69, 83.22)
}

# Dashboard heading
st.title("🌍 Air Quality Index Dashboard")
st.caption(
    "Historical AQI Analysis | PySpark | K-Means | Interactive Visualization"
)

st.info(
    "This dashboard analyses historical AQI data. "
    "It does not provide live AQI readings or future predictions."
)

# Sidebar filters
st.sidebar.header("Dashboard Filters")

cities = sorted(city["City"].dropna().unique())
selected = st.sidebar.selectbox("Select City", cities)

row = city[city["City"] == selected].iloc[0]

# Summary metrics
st.subheader("📊 City Overview")

avg_all = city["AQI"].mean()
rank = int(
    city["AQI"].rank(ascending=False)[
        city["City"] == selected
    ].iloc[0]
)

a, b, c, d = st.columns(4)

a.metric(
    "Average AQI",
    round(row["AQI"], 1),
    delta=round(row["AQI"] - avg_all, 1),
    delta_color="inverse"
)

b.metric("Average PM2.5", round(row["PM25"], 1))
c.metric("Pollution Cluster", row["Cluster"])
d.metric("Pollution Rank", f"{rank} of {len(city)}")

category = aqi_category(row["AQI"])

st.info(f"{selected} has an average AQI category of **{category}**.")
st.warning(advice[category])

st.divider()

# City-wise AQI chart
st.subheader("🏙️ Average AQI by City")

chart = city.sort_values("AQI", ascending=False)

fig = px.bar(
    chart,
    x="City",
    y="AQI",
    color="AQI",
    title="City-wise Average AQI",
    labels={"AQI": "Average AQI"},
    color_continuous_scale="RdYlGn_r"
)

fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Most polluted and cleanest cities
t1, t2 = st.columns(2)

with t1:
    st.subheader("🔴 Most Polluted 5")
    st.dataframe(
        city.nlargest(5, "AQI")[["City", "AQI"]].round(1),
        hide_index=True,
        use_container_width=True
    )

with t2:
    st.subheader("🟢 Cleanest 5")
    st.dataframe(
        city.nsmallest(5, "AQI")[["City", "AQI"]].round(1),
        hide_index=True,
        use_container_width=True
    )

# India map
st.subheader("🗺️ AQI Map of India")

map_df = city.copy()

map_df["lat"] = map_df["City"].map(
    lambda c: coords.get(c, (None, None))[0]
)
map_df["lon"] = map_df["City"].map(
    lambda c: coords.get(c, (None, None))[1]
)

map_df = map_df.dropna(subset=["lat", "lon"])

fig = px.scatter_geo(
    map_df,
    lat="lat",
    lon="lon",
    size="AQI",
    color="AQI",
    hover_name="City",
    color_continuous_scale="RdYlGn_r",
    scope="asia",
    title="City-wise Average AQI"
)

fig.update_geos(
    center=dict(lat=22, lon=80),
    projection_scale=4
)

st.plotly_chart(fig, use_container_width=True)

# Pollutant profile
st.subheader("🧪 Pollutant Profile")

pollutants = ["PM25", "PM10", "NO2", "CO", "SO2", "O3"]

compare = pd.DataFrame({
    "Pollutant": pollutants,
    selected: row[pollutants].values,
    "All-city average": city[pollutants].mean().values
}).melt(
    id_vars="Pollutant",
    var_name="Type",
    value_name="Value"
)

fig = px.bar(
    compare,
    x="Pollutant",
    y="Value",
    color="Type",
    barmode="group",
    title=f"{selected} vs All-city Average"
)

st.plotly_chart(fig, use_container_width=True)

# Yearly and monthly trends
left, right = st.columns(2)

with left:
    st.subheader("📅 Yearly AQI Trend")

    data = year[year["City"] == selected].sort_values("Year")

    fig = px.line(
        data,
        x="Year",
        y="AQI",
        markers=True,
        title=f"Yearly Trend - {selected}"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📆 Monthly AQI Trend")

    fig = px.line(
        month,
        x="Month",
        y="AQI",
        markers=True,
        title="Average AQI by Month"
    )

    st.plotly_chart(fig, use_container_width=True)

# Compare cities
st.subheader("🆚 Compare Cities")

picked = st.multiselect(
    "Choose cities to compare",
    cities,
    default=[selected]
)

yr_min = int(year["Year"].min())
yr_max = int(year["Year"].max())

yr = st.slider(
    "Select year range",
    min_value=yr_min,
    max_value=yr_max,
    value=(yr_min, yr_max)
)

if picked:
    data = year[
        (year["City"].isin(picked))
        & (year["Year"].between(yr[0], yr[1]))
    ].sort_values("Year")

    fig = px.line(
        data,
        x="Year",
        y="AQI",
        color="City",
        markers=True,
        title="Yearly AQI Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

st.caption("2020 data ends on 1 July, so it is a partial year.")

# Heatmap
st.subheader("🔥 AQI Heatmap (City vs Year)")

pivot = year.pivot(
    index="City",
    columns="Year",
    values="AQI"
)

fig = px.imshow(
    pivot,
    aspect="auto",
    color_continuous_scale="RdYlGn_r",
    title="Average AQI by City and Year"
)

st.plotly_chart(fig, use_container_width=True)

# Pollution cluster graph
st.subheader("🔬 Pollution Clusters")

fig = px.scatter(
    city,
    x="PM25",
    y="PM10",
    color="Cluster",
    hover_name="City",
    title="City Pollution Profile",
    labels={
        "PM25": "Average PM2.5",
        "PM10": "Average PM10"
    }
)

st.plotly_chart(fig, use_container_width=True)

# K-Means evaluation
st.divider()
st.subheader("📈 K-Means Cluster Evaluation")

try:
    X = city[features].dropna()
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)

    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        score = silhouette_score(X_scaled, labels)

        a, b = st.columns(2)
        a.metric("Silhouette Score", round(score, 3))

        if hasattr(model, "inertia_"):
            b.metric("K-Means Inertia", round(model.inertia_, 2))

        st.caption(
            "Silhouette score evaluates how well clusters are separated. "
            "It is not prediction accuracy."
        )
    else:
        st.info(
            "Silhouette score cannot be calculated because "
            "the data does not contain enough distinct clusters."
        )

except Exception as e:
    st.info(f"Cluster evaluation unavailable: {e}")

# Cluster Finder
st.subheader("🎯 Cluster Finder")
st.caption(
    "Enter pollutant values to identify the closest learned pollution group."
)

cols = st.columns(len(features))
values = []

for input_col, feature in zip(cols, features):
    values.append(
        input_col.number_input(
            feature,
            value=float(round(city[feature].mean(), 1))
        )
    )

if st.button("Find Cluster"):
    X_input = pd.DataFrame([values], columns=features)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]

    matching = city[city["Cluster_ID"] == pred]["Cluster"]

    if not matching.empty:
        st.success(f"Predicted pollution cluster: **{matching.iloc[0]}**")
    else:
        st.success(f"Cluster ID: {pred}")

# AQI category distribution
st.subheader("AQI Category Distribution")

fig = px.bar(
    bucket,
    x="AQI_Bucket",
    y="Days",
    color="AQI_Bucket",
    title="Records by AQI Category",
    labels={"Days": "Number of Records"}
)

st.plotly_chart(fig, use_container_width=True)

fig = px.pie(
    bucket,
    names="AQI_Bucket",
    values="Days",
    hole=0.5,
    title="Share of Records by AQI Category"
)

st.plotly_chart(fig, use_container_width=True)

# Interactive AQI Risk Explorer
st.divider()
st.subheader("🚨 Interactive AQI Risk Explorer")

threshold = st.slider(
    "Set average AQI alert threshold",
    min_value=50,
    max_value=500,
    value=200,
    step=10
)

risk_cities = city[city["AQI"] >= threshold].copy()
risk_cities = risk_cities.sort_values("AQI", ascending=False)

st.metric(
    "Cities Above Selected Threshold",
    len(risk_cities)
)

if not risk_cities.empty:
    st.warning(
        f"{len(risk_cities)} cities have an average AQI "
        f"of {threshold} or higher in this dataset."
    )

    st.dataframe(
        risk_cities[["City", "AQI"]].round(2),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download High-Risk Cities CSV",
        risk_cities[["City", "AQI"]].to_csv(index=False),
        "high_risk_cities.csv",
        "text/csv"
    )
else:
    st.success("No cities meet the selected threshold.")

# View data and download
with st.expander("View City-wise Analysis Data"):
    st.dataframe(city, use_container_width=True)

st.download_button(
    "⬇️ Download City Data",
    city.to_csv(index=False),
    "city_summary.csv",
    "text/csv"
)

st.caption(
    "Historical data only. Risk thresholds apply to average historical AQI. "
    "K-Means provides exploratory grouping, not future AQI prediction."
)
