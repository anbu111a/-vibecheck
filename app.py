import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
import pandas as pd

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="VibeCheck SCADA", page_icon="⚡", layout="wide")

# 2. Custom CSS for an Industrial UI
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #1E2130;
        border: 1px solid #39446B;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        border-left: 5px solid #00F0FF;
    }
    /* Headers and Text */
    h1, h2, h3 {
        color: #E2E8F0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 3. Dashboard Header
st.title("⚡ VibeCheck: Edge Telemetry Dashboard")
st.markdown("**Status:** `ONLINE` | **Location:** `SRMIST Edge Node 01` | **Protocol:** `MQTT Secured`")
st.markdown("---")

# 4. Sidebar Controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/3/3a/SRM_Institute_of_Science_and_Technology_logo.png", width=150)
st.sidebar.header("🎛️ Diagnostics Control")
motor_speed = st.sidebar.slider("Rotor Frequency (Hz)", 1.0, 10.0, 5.0)
noise_level = st.sidebar.slider("Ambient Sensor Noise", 0.05, 0.5, 0.2)
contamination_rate = st.sidebar.slider("ML Sensitivity (Contamination)", 0.001, 0.02, 0.005, format="%.3f")

if st.sidebar.button("🚀 Initialize Telemetry Scan", type="primary", use_container_width=True):
    
    # 5. Simulate Telemetry & Inject Faults
    np.random.seed(42)
    time = np.linspace(0, 10, 1000)
    baseline_signal = np.sin(2 * np.pi * motor_speed * time)
    noise = np.random.normal(0, noise_level, size=time.shape)
    vibration_data = baseline_signal + noise

    anomaly_indices = [150, 420, 710, 880]
    for idx in anomaly_indices:
        vibration_data[idx] += np.random.choice([-4.5, 4.5])

    # 6. ML Anomaly Detection
    X = vibration_data.reshape(-1, 1)
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    predictions = model.fit_predict(X)
    detected_indices = np.where(predictions == -1)[0]

    # 7. KPI Metrics (Top Row)
    col1, col2, col3 = st.columns(3)
    col1.metric("Live Data Packets", f"{len(vibration_data)} pts", "Nominal")
    col2.metric("Injected Hardware Faults", len(anomaly_indices), "-Critical", delta_color="inverse")
    col3.metric("Isolation Forest Detections", len(detected_indices), "Anomalies Flagged", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # 8. Interactive Plotly Graph
    fig = go.Figure()
    # Normal Signal Line
    fig.add_trace(go.Scatter(
        x=time, y=vibration_data, mode='lines', 
        name='Steady-State Vibration',
        line=dict(color='#00F0FF', width=1.5)
    ))
    # Detected Fault Markers
    fig.add_trace(go.Scatter(
        x=time[detected_indices], y=vibration_data[detected_indices], 
        mode='markers', name='ML Detected Fault',
        marker=dict(color='#FF003C', size=12, symbol='x', line=dict(color='white', width=1))
    ))
    
    fig.update_layout(
        title="High-Frequency Vibration Spectrum (Interactive)",
        template="plotly_dark",
        plot_bgcolor='rgba(14, 17, 23, 1)',
        paper_bgcolor='rgba(14, 17, 23, 1)',
        xaxis_title="Time (Seconds)",
        yaxis_title="G-Force Amplitude",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 9. Dynamic Data Table
    st.subheader("📋 Autonomous SCADA Action Logs")
    log_data = []
    for idx in detected_indices:
        g_force = vibration_data[idx]
        status = "🔴 SEVERE SHOCK" if abs(g_force) > 3.0 else "🟡 MINOR OUTLIER"
        action = "Motor Trip Initiated" if abs(g_force) > 3.0 else "Warning Registered"
        log_data.append({
            "Timestamp (s)": f"{time[idx]:.3f}",
            "Peak Amplitude (G)": f"{g_force:.2f}",
            "Diagnostic Status": status,
            "System Action": action
        })
    
    df_logs = pd.DataFrame(log_data)
    st.dataframe(df_logs, use_container_width=True)

    csv = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Edge Telemetry (CSV)",
        data=csv,
        file_name='vibecheck_industrial_logs.csv',
        mime='text/csv',
    )
else:
    st.info("📡 **System Standing By:** Adjust parameters in the sidebar and click **Initialize Telemetry Scan** to begin monitoring.")
