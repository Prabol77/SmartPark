import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from computer_vision.parking_pipeline import ParkingPipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

SEGMENTATION_MODEL = (
    ROOT
    / "runs"
    / "parking_segmentation_adaptation"
    / "ufpr04_finetune_v1"
    / "weights"
    / "best.pt"
)

OCCUPANCY_MODEL = (
    ROOT / "models" / "best_occupancy_ufpr04_holdout.pth"
)

st.set_page_config(
    page_title="SmartPark | Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. VISUAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --sp-bg: #0b1120;
        --sp-panel: #111b2e;
        --sp-panel-2: #162238;
        --sp-border: rgba(148, 163, 184, .16);
        --sp-muted: #94a3b8;
        --sp-text: #e8eef8;
        --sp-blue: #60a5fa;
        --sp-green: #34d399;
        --sp-red: #fb7185;
        --sp-amber: #fbbf24;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(37, 99, 235, .12), transparent 34%),
            var(--sp-bg);
        color: var(--sp-text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, .82);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.025em;
    }

    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #15233a, #101a2b);
        border: 1px solid var(--sp-border);
        border-radius: 16px;
        padding: 17px 18px;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }

    [data-testid="stMetricLabel"] {
        color: #a9b8cc !important;
        font-size: .82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        font-size: .76rem !important;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(17, 27, 46, .78);
        border: 1px dashed rgba(96, 165, 250, .48);
        border-radius: 15px;
    }

    .sp-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 1.2rem 0;
    }

    .sp-logo {
        display: flex;
        width: 43px;
        height: 43px;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        color: white;
        font-size: 1.35rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, .28);
    }

    .sp-brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }

    .sp-brand-sub {
        color: var(--sp-muted);
        font-size: .74rem;
        margin-top: 3px;
    }

    .sp-eyebrow {
        color: #7db7ff;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .68rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }

    .sp-subtitle {
        color: var(--sp-muted);
        font-size: .91rem;
        margin-top: -.5rem;
    }

    .sp-panel {
        background: linear-gradient(145deg, rgba(21, 34, 56, .96), rgba(15, 25, 42, .96));
        border: 1px solid var(--sp-border);
        border-radius: 17px;
        padding: 18px 20px;
    }

    .sp-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 10px;
        border: 1px solid rgba(52, 211, 153, .25);
        border-radius: 999px;
        background: rgba(16, 185, 129, .10);
        color: #6ee7b7;
        font-size: .75rem;
        font-weight: 600;
    }

    .sp-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 9px rgba(52, 211, 153, .8);
    }

    .sp-muted {
        color: var(--sp-muted);
        font-size: .82rem;
    }

    .sp-slot-id {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #eaf2ff;
    }

    .sp-footer {
        border-top: 1px solid var(--sp-border);
        margin-top: 2rem;
        padding-top: 1rem;
        color: #718198;
        font-size: .76rem;
        text-align: center;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #0ea5e9);
        border: 0;
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
        box-shadow: 0 7px 20px rgba(37, 99, 235, .2);
    }

    div.stButton > button {
        border-radius: 10px;
    }

    hr {
        border-color: var(--sp-border) !important;
    }

    [data-testid="stSidebar"] {
        background: #0e192b;
        border-right: 1px solid var(--sp-border);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--sp-border);
        border-radius: 13px;
        background: rgba(17, 27, 46, .55);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563eb, #34d399);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_parking_pipeline():
    missing = [
        str(path)
        for path in (SEGMENTATION_MODEL, OCCUPANCY_MODEL)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model checkpoint(s) not found:\n" + "\n".join(missing)
        )

    return ParkingPipeline(
        segmentation_checkpoint=SEGMENTATION_MODEL,
        occupancy_checkpoint=OCCUPANCY_MODEL,
        yolo_confidence=0.15,
        yolo_iou=0.50,
    )


# ============================================================
# 4. HEADER AND SIDEBAR
# ============================================================

st.markdown(
    """
    <div class="sp-brand">
        <div class="sp-logo">P</div>
        <div>
            <div class="sp-brand-name">SmartPark</div>
            <div class="sp-brand-sub">AI parking intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

head_left, head_right = st.columns([3, 1])
with head_left:
    st.markdown('<div class="sp-eyebrow">CSE472 · Intelligent Parking System</div>', unsafe_allow_html=True)
    st.title("Parking Intelligence Dashboard")
    st.markdown(
        '<div class="sp-subtitle">Analyse parking occupancy, inspect detected spaces and review uncertain predictions.</div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        '<div style="text-align:right;padding-top:14px"><span class="sp-status"><span class="sp-dot"></span>Inference engine ready</span></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## 🅿️ SmartPark")
    st.caption("Intelligent Parking Management")
    st.divider()
    st.markdown("### Model stack")
    st.markdown("- **Segmentation:** YOLOv8n-seg")
    st.markdown("- **Classification:** CNN")
    st.markdown("- **Inference:** CUDA when available")
    st.markdown("- **Input:** Uploaded image")
    st.divider()
    st.markdown("### Confidence guide")
    st.markdown("🟢 **High** · 95% and above")
    st.markdown("🟡 **Review** · 80% to below 95%")
    st.markdown("🟠 **Uncertain** · Below 80%")
    st.divider()
    st.caption("Live CCTV integration is not enabled.")
    if st.button("Clear current analysis", use_container_width=True):
        for key in (
            "analysis_hash", "predictions", "annotated_image",
            "source_filename", "detection_count", "inference_device",
            "analysis_time",
        ):
            st.session_state.pop(key, None)
        st.rerun()


# ============================================================
# 5. UPLOAD AND ANALYSE
# ============================================================

st.markdown("### Upload parking image")
st.markdown(
    '<div class="sp-muted">Use a clear elevated or top-down view with visible parking-bay markings for the best results.</div>',
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([1.25, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose a parking-lot image",
        type=["jpg", "jpeg", "png"],
        key="smartpark_main_uploader",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        needs_analysis = st.session_state.get("analysis_hash") != image_hash

        if st.button(
            "🚀  Analyse Parking Lot" if needs_analysis else "↻  Re-analyse Image",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("YOLO is segmenting parking spaces and the CNN is classifying occupancy..."):
                    pipeline = load_parking_pipeline()
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    if image_bgr is None:
                        raise ValueError("The uploaded file could not be decoded as an image.")

                    result = pipeline.analyse(image_bgr)
                    if not result.get("rows"):
                        raise ValueError("No parking spaces were successfully classified.")

                    st.session_state["analysis_hash"] = image_hash
                    st.session_state["predictions"] = result["rows"]
                    st.session_state["annotated_image"] = result["annotated_rgb"]
                    st.session_state["source_filename"] = uploaded_file.name
                    st.session_state["detection_count"] = result["detection_count"]
                    st.session_state["inference_device"] = result["device"]
                    st.session_state["analysis_time"] = datetime.now()
                st.success("Analysis completed successfully.")
                st.rerun()
            except Exception as error:
                st.error(f"Parking analysis failed: {error}")

with preview_col:
    if uploaded_file is not None:
        st.image(uploaded_file, caption=f"Input image · {uploaded_file.name}", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="sp-panel" style="min-height:155px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center">
                <div style="font-size:2rem">▧</div>
                <div style="font-weight:700;color:#e8eef8">Your parking image preview</div>
                <div class="sp-muted">JPG · JPEG · PNG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()


# ============================================================
# 6. REQUIRE A SUCCESSFUL ANALYSIS
# ============================================================

predictions = st.session_state.get("predictions")
if not predictions:
    st.info("Upload a parking image and select **Analyse Parking Lot** to populate the dashboard.")
    st.markdown(
        """
        <div class="sp-panel">
            <div class="sp-eyebrow">Analysis workflow</div>
            <b>01 · Upload</b> your parking-lot image<br>
            <b>02 · Segment</b> parking-space candidates with YOLOv8n-seg<br>
            <b>03 · Classify</b> each extracted crop with the CNN<br>
            <b>04 · Review</b> occupancy metrics, confidence and exported results
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

annotated_image = st.session_state["annotated_image"]
source_filename = st.session_state["source_filename"]
detection_count = st.session_state["detection_count"]
inference_device = st.session_state["inference_device"]
last_updated = st.session_state["analysis_time"]

total_slots = len(predictions)
occupied = sum(row["prediction"] == "OCCUPIED" for row in predictions)
available = sum(row["prediction"] == "EMPTY" for row in predictions)
occupancy_rate = occupied / total_slots * 100 if total_slots else 0

high_confidence = sum(row["confidence_status"] == "HIGH_CONFIDENCE" for row in predictions)
review_count = sum(row["confidence_status"] == "REVIEW" for row in predictions)
uncertain_count = sum(row["confidence_status"] == "UNCERTAIN" for row in predictions)
mean_confidence = (
    sum(float(row["confidence_percent"]) for row in predictions) / total_slots
    if total_slots else 0
)


# ============================================================
# 7. PARKING OVERVIEW
# ============================================================

st.markdown("## Parking overview")
st.markdown(
    f'<div class="sp-muted">Results for <b>{source_filename}</b> · '
    f'{last_updated.strftime("%d %b %Y, %I:%M %p")} · '
    f'Inference device: {inference_device.upper()}</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Classified spaces", f"{total_slots}", help="Successfully classified detections, not guaranteed total physical capacity.")
m2.metric("Available", f"{available}", f"{available / total_slots * 100:.1f}% of classified" if total_slots else None)
m3.metric("Occupied", f"{occupied}", f"{occupied / total_slots * 100:.1f}% of classified" if total_slots else None)
m4.metric("Occupancy rate", f"{occupancy_rate:.1f}%")

st.markdown("#### Prediction confidence")
c1, c2, c3, c4 = st.columns(4)
c1.metric("High confidence", high_confidence)
c2.metric("Review required", review_count)
c3.metric("Uncertain", uncertain_count)
c4.metric("Mean CNN score", f"{mean_confidence:.1f}%")

st.progress(
    high_confidence / total_slots if total_slots else 0,
    text=f"{high_confidence} of {total_slots} predictions are in the high-confidence band",
)
st.caption("Confidence is the model's prediction score; it is not independently verified classification accuracy.")
st.divider()


# ============================================================
# 8. ANNOTATED IMAGE AND SUMMARY
# ============================================================

st.markdown("## Parking lot visualisation")
image_col, summary_col = st.columns([2.2, 1], gap="large")

with image_col:
    st.image(
        annotated_image,
        caption=f"YOLOv8n-seg masks with CNN occupancy labels · {source_filename}",
        use_container_width=True,
    )

with summary_col:
    st.markdown("#### Detection summary")
    st.metric("YOLO detections", detection_count)
    st.metric("Successfully classified", total_slots)
    st.markdown("#### Occupancy mix")
    st.write(f"🟢 Available: **{available}**")
    st.write(f"🔴 Occupied: **{occupied}**")
    st.markdown("#### Confidence bands")
    st.write(f"🟢 High: **{high_confidence}**")
    st.write(f"🟡 Review: **{review_count}**")
    st.write(f"🟠 Uncertain: **{uncertain_count}**")
    st.caption("Only successfully classified detections contribute to the occupancy rate.")

st.divider()


# ============================================================
# 9. ANALYTICS
# ============================================================

st.markdown("## Parking analytics")
chart_left, chart_right = st.columns(2, gap="large")

with chart_left:
    st.markdown("#### Occupancy distribution")
    occupancy_chart = go.Figure(
        data=[
            go.Pie(
                labels=["Available", "Occupied"],
                values=[available, occupied],
                hole=0.68,
                marker=dict(colors=["#34d399", "#fb7185"], line=dict(color="#111827", width=3)),
                textinfo="label+percent",
                textfont=dict(color="#e5edf8", size=12),
                sort=False,
            )
        ]
    )
    occupancy_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(t=15, b=15, l=10, r=10),
        legend=dict(orientation="h", y=-0.08),
        font=dict(family="DM Sans, sans-serif", color="#dbe7f5"),
    )
    st.plotly_chart(occupancy_chart, use_container_width=True)

with chart_right:
    st.markdown("#### Confidence distribution")
    confidence_chart = go.Figure(
        data=[
            go.Bar(
                x=["High", "Review", "Uncertain"],
                y=[high_confidence, review_count, uncertain_count],
                marker_color=["#34d399", "#fbbf24", "#fb923c"],
                text=[high_confidence, review_count, uncertain_count],
                textposition="outside",
                cliponaxis=False,
            )
        ]
    )
    confidence_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(t=25, b=20, l=15, r=15),
        yaxis_title="Number of classified spaces",
        showlegend=False,
        font=dict(family="DM Sans, sans-serif", color="#dbe7f5"),
    )
    confidence_chart.update_yaxes(rangemode="tozero", gridcolor="rgba(148,163,184,.12)")
    st.plotly_chart(confidence_chart, use_container_width=True)

st.divider()


# ============================================================
# 10. SLOT MONITORING
# ============================================================

st.markdown("## Individual parking spaces")
filter_col, search_col = st.columns([1, 1])
with filter_col:
    filter_option = st.selectbox(
        "Filter spaces",
        ["All spaces", "Available", "Occupied", "High confidence", "Review required", "Uncertain"],
    )
with search_col:
    search_slot = st.text_input("Find slot ID", placeholder="e.g. P001")

filtered = list(predictions)

if filter_option == "Available":
    filtered = [row for row in filtered if row["prediction"] == "EMPTY"]
elif filter_option == "Occupied":
    filtered = [row for row in filtered if row["prediction"] == "OCCUPIED"]
elif filter_option == "High confidence":
    filtered = [row for row in filtered if row["confidence_status"] == "HIGH_CONFIDENCE"]
elif filter_option == "Review required":
    filtered = [row for row in filtered if row["confidence_status"] == "REVIEW"]
elif filter_option == "Uncertain":
    filtered = [row for row in filtered if row["confidence_status"] == "UNCERTAIN"]

if search_slot.strip():
    filtered = [
        row for row in filtered
        if search_slot.strip().lower() in str(row["slot_id"]).lower()
    ]

st.caption(f"Showing {len(filtered)} of {total_slots} classified spaces")

if not filtered:
    st.info("No parking spaces match the selected filter.")
else:
    for start in range(0, len(filtered), 4):
        cols = st.columns(4)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(filtered):
                break
            slot = filtered[idx]
            is_occupied = slot["prediction"] == "OCCUPIED"
            status_color = "#fb7185" if is_occupied else "#34d399"
            confidence = float(slot["confidence_percent"])
            with col:
                st.markdown(
                    f"""
                    <div class="sp-panel" style="padding:15px;margin-bottom:10px;border-top:3px solid {status_color}">
                        <div class="sp-muted">PARKING CANDIDATE</div>
                        <div class="sp-slot-id">{slot['slot_id']}</div>
                        <div style="font-size:1.05rem;font-weight:700;color:{status_color};margin:8px 0">
                            {'OCCUPIED' if is_occupied else 'AVAILABLE'}
                        </div>
                        <div class="sp-muted">CNN score · {confidence:.2f}%</div>
                        <div style="height:5px;background:#26364e;border-radius:9px;margin-top:8px;overflow:hidden">
                            <div style="height:5px;width:{max(0,min(100,confidence)):.1f}%;background:{status_color};border-radius:9px"></div>
                        </div>
                        <div class="sp-muted" style="margin-top:8px">{slot['confidence_status'].replace('_', ' ')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.divider()


# ============================================================
# 11. AI REVIEW CENTRE
# ============================================================

st.markdown("## AI review centre")
review_slots = [
    row for row in predictions
    if row["confidence_status"] in ("REVIEW", "UNCERTAIN")
]

if review_slots:
    st.warning(f"{len(review_slots)} candidate(s) have lower-confidence predictions and should be manually checked.")

    review_rows = []
    for row in review_slots:
        review_rows.append({
            "Slot ID": row["slot_id"],
            "Prediction": row["prediction"],
            "CNN score": f"{float(row['confidence_percent']):.2f}%",
            "Review band": row["confidence_status"].replace("_", " "),
        })

    # Render review rows natively without pandas or PyArrow dependencies
    for item in review_rows:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)

            col1.markdown("**Slot ID**")
            col1.write(item["Slot ID"])

            col2.markdown("**Prediction**")
            col2.write(item["Prediction"])

            col3.markdown("**CNN score**")
            col3.write(item["CNN score"])

            col4.markdown("**Review band**")
            col4.write(item["Review band"])
else:
    st.success("No predictions fall into the review or uncertain bands.")

st.caption("Manual verification is recommended for all review and uncertain predictions.")
st.divider()


# ============================================================
# 12. EXPORT
# ============================================================

st.markdown("## Export analysis")
st.markdown(
    '<div class="sp-muted">Download the current image analysis for documentation or further review.</div>',
    unsafe_allow_html=True,
)

csv_buffer = io.StringIO()
fieldnames = list(predictions[0].keys())
writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(predictions)

summary_buffer = io.StringIO()
summary_writer = csv.writer(summary_buffer)
summary_writer.writerow(["Metric", "Value"])
summary_writer.writerow(["Source image", source_filename])
summary_writer.writerow(["Analysis time", last_updated.isoformat(timespec="seconds")])
summary_writer.writerow(["YOLO detections", detection_count])
summary_writer.writerow(["Successfully classified", total_slots])
summary_writer.writerow(["Available predictions", available])
summary_writer.writerow(["Occupied predictions", occupied])
summary_writer.writerow(["Occupancy rate (%)", f"{occupancy_rate:.2f}"])
summary_writer.writerow(["High confidence", high_confidence])
summary_writer.writerow(["Review required", review_count])
summary_writer.writerow(["Uncertain", uncertain_count])
summary_writer.writerow(["Inference device", inference_device])

export_col1, export_col2 = st.columns(2)
with export_col1:
    st.download_button(
        "⬇ Download slot-level CSV",
        data=csv_buffer.getvalue(),
        file_name="smartpark_slot_predictions.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )
with export_col2:
    st.download_button(
        "⬇ Download analysis summary",
        data=summary_buffer.getvalue(),
        file_name="smartpark_analysis_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    """
    <div class="sp-footer">
        <b>SmartPark · CSE472</b><br>
        Intelligent Parking Management and Slot Prediction System<br>
        YOLOv8n-seg parking-space segmentation · CNN occupancy classification<br>
        Image-based inference prototype · Live CCTV integration is future work
    </div>
    """,
    unsafe_allow_html=True,
)
