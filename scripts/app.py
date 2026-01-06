import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from severity_rules import estimate_severity

# ---------------------------
# Load trained YOLO model
# ---------------------------
MODEL_PATH = "runs/detect/rdd_yolov8/weights/best.pt"
model = YOLO(MODEL_PATH)

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="AI Road Damage Detection",
    layout="centered"
)

st.title("🛣️ AI Road Damage Detection System")
st.write(
    "Upload a road image to detect **damage presence**, "
    "**damage type**, and **severity level**."
)

# ---------------------------
# Image uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Damage"):
        results = model(img_array)[0]

        # ---------------------------
        # No damage detected
        # ---------------------------
        if len(results.boxes) == 0:
            st.success("✅ Road is NOT damaged")

        # ---------------------------
        # Damage detected
        # ---------------------------
        else:
            st.warning("⚠ Road damage detected")

            st.subheader("Detection Summary")
            st.write(f"Total damaged regions detected: **{len(results.boxes)}**")

            for i, (box, cls, conf) in enumerate(
                zip(results.boxes.xyxy,
                    results.boxes.cls,
                    results.boxes.conf)
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                damage_type = model.names[int(cls)]
                confidence = float(conf)

                severity = estimate_severity(
                    [x1, y1, x2, y2],
                    confidence,
                    damage_type,
                    img_array.shape
                )

                # Draw bounding box
                cv2.rectangle(
                    img_array,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"{damage_type} | {severity}"

                cv2.putText(
                    img_array,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    2
                )

                # Show text output
                st.write(
                    f"🔹 **Damage {i+1}:** "
                    f"Type = `{damage_type}`, "
                    f"Severity = `{severity}`"
                )

            st.image(
                img_array,
                caption="Detection Result",
                use_container_width=True
            )
