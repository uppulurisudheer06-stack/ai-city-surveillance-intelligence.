import streamlit as st
from pathlib import Path
from ultralytics import YOLO
import cv2
import tempfile
import os
from collections import Counter


# ============================================================
# SMART CITY SHARED DATA
# ============================================================

if "city_metrics" not in st.session_state:
    st.session_state.city_metrics = {
        "cars": 0,
        "motorcycles": 0,
        "buses": 0,
        "trucks": 0,
        "total_vehicles": 0,
        "source": "No detection yet"
    }


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI City Surveillance",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "yolo11n.pt"
EMERGENCY_MODEL_PATH = BASE_DIR / "runs" / "detect" / "models" / "emergency_vehicle-3" / "weights" / "best.pt"
FIRE_SMOKE_MODEL_PATH = BASE_DIR / "runs" / "detect" / "models" / "fire_smoke" / "weights" / "best.pt"
PHONE_IMAGE_DIR = BASE_DIR / "phone images"


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE = 0.35
IMAGE_SIZE = 1280
IOU_THRESHOLD = 0.50


# ============================================================
# VEHICLE CLASSES
# ============================================================

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def load_emergency_model():
    return YOLO(str(EMERGENCY_MODEL_PATH))


@st.cache_resource
def load_fire_smoke_model():
    return YOLO(str(FIRE_SMOKE_MODEL_PATH))


model = load_model()


# ============================================================
# HELPER FUNCTION:
# COUNT ALL OBJECTS
# ============================================================

def get_object_counts(result):

    counts = Counter()

    if result.boxes is None:
        return counts

    for class_id in result.boxes.cls:

        class_id = int(class_id)

        class_name = model.names[class_id]

        counts[class_name] += 1

    return counts


# ============================================================
# HELPER FUNCTION:
# COUNT VEHICLES ONLY
# ============================================================

def get_vehicle_counts(result):

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    if result.boxes is None:
        return counts

    for class_id in result.boxes.cls:

        class_id = int(class_id)

        class_name = model.names[class_id].lower()

        if class_name in counts:
            counts[class_name] += 1

    return counts


# ============================================================
# SMALL EMERGENCY VEHICLE DETECTION USING IMAGE TILES
# ============================================================

def detect_emergency_with_tiles(
    image,
    emergency_model,
    confidence=0.30,
    tile_size=640,
    overlap=0.25
):
    """
    Detect emergency vehicles on the complete image in one pass.
    This avoids duplicate detections introduced when the same
    vehicle appears in several overlapping tiles.
    The tile_size and overlap parameters are retained so existing
    calls elsewhere in app.py do not need to be changed.
    """

    if image is None:
        return image, Counter(), []

    results = emergency_model.predict(
        source=image,
        conf=confidence,
        imgsz=1280,
        iou=0.45,
        max_det=100,
        verbose=False
    )

    result = results[0]
    counts = Counter()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in coords]
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            class_name = emergency_model.names[class_id]

            counts[class_name] += 1
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": score,
                "class_id": class_id
            })

    # Use YOLO's standard annotation: one box per retained detection.
    annotated = result.plot(labels=True, conf=True, boxes=True)

    return annotated, counts, detections


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚦 AI City Surveillance")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "📱 Phone Image Detection",
        "🚑 Emergency Vehicle",
        "🔥 Fire & Smoke",
        "Alerts"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🚦 AI City Surveillance Intelligence Platform")

    st.write(
        "AI-based image and video surveillance using YOLO11."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🖼️ Image Detection",
            "Available"
        )

    with col2:
        st.metric(
            "🎥 Video Detection",
            "Available"
        )

    with col3:
        st.metric(
            "🤖 YOLO Model",
            "YOLO11n"
        )

    st.markdown("---")

    st.subheader("System Workflow")

    st.write(
        """
        📷 Image / 🎥 Video
        ↓
        YOLO11
        ↓
        Object Detection
        ↓
        Bounding Boxes
        ↓
        Object Classification
        ↓
        Vehicle Analytics
        ↓
        Surveillance Dashboard
        """
    )

    st.markdown("---")

    st.subheader("Current Detection Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence",
            CONFIDENCE
        )

    with col2:
        st.metric(
            "Image Size",
            IMAGE_SIZE
        )

    with col3:
        st.metric(
            "IoU",
            IOU_THRESHOLD
        )

    st.info(
        "The system detects all object classes supported by "
        "the loaded YOLO model. Vehicle statistics are shown separately."
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

elif page == "Image Detection":

    st.title("🖼️ AI Image Object Detection")

    st.write(
        "Upload an image. YOLO11 will detect objects supported "
        "by the pretrained model."
    )

    uploaded_image = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(
                uploaded_image.read()
            )

            image_path = temp_file.name

        st.subheader("📷 Original Image")

        st.image(
            image_path,
            use_container_width=True
        )

        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------

        if st.button(
            "🔍 Detect Objects",
            key="image_detection"
        ):

            with st.spinner(
                "YOLO11 is analyzing the image..."
            ):

                results = model.predict(
                    source=image_path,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )

            result = results[0]

            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.subheader("🎯 Detection Result")

            annotated_image = result.plot(
                labels=True,
                conf=True,
                boxes=True
            )

            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

            # ------------------------------------------------
            # OBJECT COUNTS
            # ------------------------------------------------

            object_counts = get_object_counts(result)

            st.markdown("---")

            st.subheader("📊 Detected Objects")

            if len(object_counts) == 0:

                st.warning(
                    "No objects were detected."
                )

            else:

                total_objects = sum(
                    object_counts.values()
                )

                st.success(
                    f"🎯 Total Objects Detected: {total_objects}"
                )

                cols = st.columns(
                    min(len(object_counts), 4)
                )

                for index, (name, count) in enumerate(
                    object_counts.items()
                ):

                    with cols[index % len(cols)]:

                        st.metric(
                            name.title(),
                            count
                        )

            # ------------------------------------------------
            # VEHICLE ANALYTICS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("🚗 Vehicle Analysis")

            vehicle_counts = get_vehicle_counts(
                result
            )

            total_vehicles = sum(
                vehicle_counts.values()
            )

            # =================================================
            # UPDATE SMART CITY DATA
            # =================================================

            st.session_state.city_metrics.update({
                "cars": vehicle_counts["car"],
                "motorcycles": vehicle_counts["motorcycle"],
                "buses": vehicle_counts["bus"],
                "trucks": vehicle_counts["truck"],
                "total_vehicles": total_vehicles,
                "source": "Image Detection"
            })

            st.success(
                f"🚗 Total Vehicles: {total_vehicles}"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🚗 Cars",
                    vehicle_counts["car"]
                )

            with col2:
                st.metric(
                    "🏍️ Motorcycles",
                    vehicle_counts["motorcycle"]
                )

            with col3:
                st.metric(
                    "🚌 Buses",
                    vehicle_counts["bus"]
                )

            with col4:
                st.metric(
                    "🚚 Trucks",
                    vehicle_counts["truck"]
                )

        # ----------------------------------------------------
        # DELETE TEMP IMAGE
        # ----------------------------------------------------

        try:
            os.remove(image_path)
        except:
            pass


# ============================================================
# VIDEO DETECTION
# ============================================================

elif page == "Video Detection":

    st.title("🎥 AI Video Object Detection")

    st.write(
        "Upload a video. YOLO11 will analyze every frame "
        "and detect supported objects."
    )

    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # ----------------------------------------------------
        # SAVE INPUT VIDEO
        # ----------------------------------------------------

        input_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_video.write(
            uploaded_video.read()
        )

        input_video.close()

        st.subheader("🎥 Original Video")

        st.video(
            input_video.name
        )

        # ----------------------------------------------------
        # PROCESS VIDEO
        # ----------------------------------------------------

        if st.button(
            "🔍 Detect Objects in Video",
            key="video_detection"
        ):

            cap = cv2.VideoCapture(
                input_video.name
            )

            if not cap.isOpened():

                st.error(
                    "❌ Unable to open the video."
                )

            else:

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                if fps <= 0:
                    fps = 30

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                # ------------------------------------------------
                # OUTPUT VIDEO
                # ------------------------------------------------

                output_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_video.close()

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_video.name,
                    fourcc,
                    fps,
                    (width, height)
                )

                progress_bar = st.progress(0)

                status_text = st.empty()

                # Maximum vehicles visible in one frame
                max_vehicle_count = 0

                # Maximum class counts seen in a frame
                max_vehicle_counts = {
                    "car": 0,
                    "motorcycle": 0,
                    "bus": 0,
                    "truck": 0
                }

                # Overall object classes observed
                objects_seen = set()

                frame_number = 0

                # ------------------------------------------------
                # FRAME-BY-FRAME PROCESSING
                # ------------------------------------------------

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    # ------------------------------------------------
                    # YOLO DETECTION
                    # ------------------------------------------------

                    results = model.predict(
                        source=frame,
                        conf=CONFIDENCE,
                        imgsz=IMAGE_SIZE,
                        iou=IOU_THRESHOLD,
                        verbose=False
                    )

                    result = results[0]

                    # ------------------------------------------------
                    # OBJECT COUNTS
                    # ------------------------------------------------

                    object_counts = get_object_counts(
                        result
                    )

                    for object_name in object_counts:

                        objects_seen.add(
                            object_name
                        )

                    # ------------------------------------------------
                    # VEHICLE COUNTS
                    # ------------------------------------------------

                    vehicle_counts = get_vehicle_counts(
                        result
                    )

                    current_vehicle_count = sum(
                        vehicle_counts.values()
                    )

                    if current_vehicle_count > max_vehicle_count:

                        max_vehicle_count = (
                            current_vehicle_count
                        )

                    for vehicle_type in max_vehicle_counts:

                        if (
                            vehicle_counts[vehicle_type]
                            >
                            max_vehicle_counts[vehicle_type]
                        ):

                            max_vehicle_counts[
                                vehicle_type
                            ] = vehicle_counts[
                                vehicle_type
                            ]

                    # ------------------------------------------------
                    # DRAW DETECTIONS
                    # ------------------------------------------------

                    annotated_frame = result.plot(
                        labels=True,
                        conf=True,
                        boxes=True
                    )

                    # ------------------------------------------------
                    # DISPLAY CURRENT FRAME VEHICLE COUNT
                    # ------------------------------------------------

                    cv2.putText(
                        annotated_frame,
                        f"Vehicles: {current_vehicle_count}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

                    # ------------------------------------------------
                    # WRITE OUTPUT
                    # ------------------------------------------------

                    writer.write(
                        annotated_frame
                    )

                    # ------------------------------------------------
                    # PROGRESS
                    # ------------------------------------------------

                    if total_frames > 0:

                        progress = (
                            frame_number /
                            total_frames
                        )

                        progress_bar.progress(
                            min(progress, 1.0)
                        )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_number}/{total_frames}"
                    )

                # ------------------------------------------------
                # RELEASE
                # ------------------------------------------------

                cap.release()

                writer.release()

                progress_bar.progress(1.0)

                status_text.success(
                    "✅ Video processing completed!"
                )

                # =================================================
                # UPDATE SMART CITY DATA
                # =================================================

                video_total_vehicles = (
                    max_vehicle_counts["car"]
                    +
                    max_vehicle_counts["motorcycle"]
                    +
                    max_vehicle_counts["bus"]
                    +
                    max_vehicle_counts["truck"]
                )

                st.session_state.city_metrics.update({
                    "cars": max_vehicle_counts["car"],
                    "motorcycles": max_vehicle_counts["motorcycle"],
                    "buses": max_vehicle_counts["bus"],
                    "trucks": max_vehicle_counts["truck"],
                    "total_vehicles": video_total_vehicles,
                    "source": "Video Detection"
                })

                # ------------------------------------------------
                # SHOW VIDEO
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "🎯 Processed Video"
                )

                st.video(
                    output_video.name
                )

                # ------------------------------------------------
                # VEHICLE STATISTICS
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "🚗 Vehicle Statistics"
                )

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "🚗 Cars",
                        max_vehicle_counts["car"]
                    )

                with col2:
                    st.metric(
                        "🏍️ Motorcycles",
                        max_vehicle_counts["motorcycle"]
                    )

                with col3:
                    st.metric(
                        "🚌 Buses",
                        max_vehicle_counts["bus"]
                    )

                with col4:
                    st.metric(
                        "🚚 Trucks",
                        max_vehicle_counts["truck"]
                    )

                with col5:
                    st.metric(
                        "🚦 Max Vehicles",
                        max_vehicle_count
                    )

                # ------------------------------------------------
                # OBJECTS SEEN IN VIDEO
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📊 Object Classes Detected in Video"
                )

                if objects_seen:

                    object_list = sorted(
                        objects_seen
                    )

                    st.write(
                        ", ".join(
                            name.title()
                            for name in object_list
                        )
                    )

                else:

                    st.warning(
                        "No objects were detected."
                    )

                st.info(
                    "The maximum vehicle count represents "
                    "the highest number of vehicles visible "
                    "simultaneously in one frame. Unique "
                    "vehicle tracking will be added later."
                )


# ============================================================
# PHONE IMAGE DETECTION
# ============================================================

elif page == "📱 Phone Image Detection":

    st.title("📱 Phone Image Object Detection")

    st.write(
        "Test YOLO11 on images captured using a mobile phone."
    )

    phone_image_dir = BASE_DIR / "phone images"

    if not phone_image_dir.exists():

        st.error(
            "❌ 'phone images' folder was not found."
        )

        st.info(
            "Create a folder named 'phone images' "
            "inside your project folder."
        )

    else:

        image_files = []

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG"
        ]:

            image_files.extend(
                phone_image_dir.glob(
                    extension
                )
            )

        if not image_files:

            st.warning(
                "No phone images found."
            )

        else:

            st.success(
                f"📱 {len(image_files)} phone images found."
            )

            selected_image = st.selectbox(
                "Choose a phone image",
                image_files,
                format_func=lambda x: x.name
            )

            st.subheader(
                "📷 Original Image"
            )

            st.image(
                str(selected_image),
                use_container_width=True
            )

            if st.button(
                "🔍 Detect Objects",
                key="phone_detection"
            ):

                with st.spinner(
                    "YOLO11 is analyzing the image..."
                ):

                    results = model.predict(
                        source=str(selected_image),
                        conf=CONFIDENCE,
                        imgsz=IMAGE_SIZE,
                        iou=IOU_THRESHOLD,
                        verbose=False
                    )

                result = results[0]

                annotated_image = result.plot(
                    labels=True,
                    conf=True,
                    boxes=True
                )

                st.subheader(
                    "🎯 Detection Result"
                )

                st.image(
                    annotated_image,
                    channels="BGR",
                    use_container_width=True
                )

                object_counts = get_object_counts(
                    result
                )

                vehicle_counts = get_vehicle_counts(
                    result
                )

                # =================================================
                # UPDATE SMART CITY DATA
                # =================================================

                phone_total_vehicles = sum(
                    vehicle_counts.values()
                )

                st.session_state.city_metrics.update({
                    "cars": vehicle_counts["car"],
                    "motorcycles": vehicle_counts["motorcycle"],
                    "buses": vehicle_counts["bus"],
                    "trucks": vehicle_counts["truck"],
                    "total_vehicles": phone_total_vehicles,
                    "source": "Phone Image Detection"
                })

                st.markdown("---")

                st.subheader(
                    "📊 Detected Objects"
                )

                for name, count in object_counts.items():

                    st.write(
                        f"**{name.title()}** : {count}"
                    )

                st.markdown("---")

                st.subheader(
                    "🚗 Vehicle Analysis"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🚗 Cars",
                        vehicle_counts["car"]
                    )

                with col2:
                    st.metric(
                        "🏍️ Motorcycles",
                        vehicle_counts["motorcycle"]
                    )

                with col3:
                    st.metric(
                        "🚌 Buses",
                        vehicle_counts["bus"]
                    )

                with col4:
                    st.metric(
                        "🚚 Trucks",
                        vehicle_counts["truck"]
                    )


# ============================================================
# FIRE & SMOKE DETECTION
# ============================================================

elif page == "🔥 Fire & Smoke":

    st.title("🔥 Fire & Smoke Detection")
    st.write("Detect Fire and Smoke using the trained YOLO11 Fire & Smoke model.")

    if not FIRE_SMOKE_MODEL_PATH.exists():
        st.error(f"❌ Fire & Smoke model not found: {FIRE_SMOKE_MODEL_PATH}")
    else:
        fire_smoke_model = load_fire_smoke_model()

        fire_smoke_confidence = st.slider(
            "Detection Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            key="fire_smoke_confidence"
        )

        fire_smoke_input_type = st.radio(
            "Choose input",
            ["🖼️ Image", "🎥 Video"],
            horizontal=True,
            key="fire_smoke_input_type"
        )

        if fire_smoke_input_type == "🖼️ Image":

            uploaded_fire_smoke_image = st.file_uploader(
                "📤 Upload Fire / Smoke Image",
                type=["jpg", "jpeg", "png"],
                key="fire_smoke_image_upload"
            )

            if uploaded_fire_smoke_image is not None:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_fire_smoke_image.name).suffix or ".jpg"
                ) as temp_file:
                    temp_file.write(uploaded_fire_smoke_image.read())
                    fire_smoke_image_path = temp_file.name

                st.subheader("📷 Original Image")
                st.image(fire_smoke_image_path, use_container_width=True)

                if st.button("🔍 Detect Fire & Smoke", key="fire_smoke_image_detection"):
                    with st.spinner("🔥 Analyzing image for fire and smoke..."):
                        results = fire_smoke_model.predict(
                            source=fire_smoke_image_path,
                            conf=fire_smoke_confidence,
                            imgsz=640,
                            iou=0.45,
                            max_det=100,
                            verbose=False
                        )

                    result = results[0]
                    annotated = result.plot(labels=True, conf=True, boxes=True)

                    st.subheader("🎯 Fire & Smoke Detection Result")
                    st.image(annotated, channels="BGR", use_container_width=True)

                    fire_count = 0
                    smoke_count = 0

                    if result.boxes is not None:
                        for class_id in result.boxes.cls:
                            class_id = int(class_id)
                            class_name = str(fire_smoke_model.names[class_id]).lower()
                            if class_name == "fire":
                                fire_count += 1
                            elif class_name == "smoke":
                                smoke_count += 1

                    total = fire_count + smoke_count

                    st.markdown("---")
                    st.subheader("📊 Fire & Smoke Statistics")

                    if total == 0:
                        st.warning("No fire or smoke detected.")
                    else:
                        st.success(f"🔥 Total Fire/Smoke Detections: {total}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔥 Fire", fire_count)
                    with col2:
                        st.metric("💨 Smoke", smoke_count)
                    with col3:
                        st.metric("🎯 Total", total)

                try:
                    os.remove(fire_smoke_image_path)
                except OSError:
                    pass

        else:

            uploaded_fire_smoke_video = st.file_uploader(
                "📤 Upload Fire / Smoke Video",
                type=["mp4", "avi", "mov", "mkv"],
                key="fire_smoke_video_upload"
            )

            if uploaded_fire_smoke_video is not None:

                input_fire_smoke_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                input_fire_smoke_video.write(uploaded_fire_smoke_video.read())
                input_fire_smoke_video.close()

                st.subheader("🎥 Original Video")
                st.video(input_fire_smoke_video.name)

                if st.button("🔍 Detect Fire & Smoke in Video", key="fire_smoke_video_detection"):
                    cap = cv2.VideoCapture(input_fire_smoke_video.name)

                    if not cap.isOpened():
                        st.error("❌ Unable to open the video.")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30

                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        output_video.close()

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_video.name, fourcc, fps, (width, height))

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        max_fire = 0
                        max_smoke = 0
                        frame_number = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_number += 1
                            results = fire_smoke_model.predict(
                                source=frame,
                                conf=fire_smoke_confidence,
                                imgsz=640,
                                iou=0.45,
                                max_det=100,
                                verbose=False
                            )
                            result = results[0]

                            current_fire = 0
                            current_smoke = 0

                            if result.boxes is not None:
                                for class_id in result.boxes.cls:
                                    class_id = int(class_id)
                                    class_name = str(fire_smoke_model.names[class_id]).lower()
                                    if class_name == "fire":
                                        current_fire += 1
                                    elif class_name == "smoke":
                                        current_smoke += 1

                            max_fire = max(max_fire, current_fire)
                            max_smoke = max(max_smoke, current_smoke)

                            annotated_frame = result.plot(labels=True, conf=True, boxes=True)
                            cv2.putText(
                                annotated_frame,
                                f"Fire: {current_fire}  Smoke: {current_smoke}",
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 0),
                                3
                            )
                            writer.write(annotated_frame)

                            if total_frames > 0:
                                progress_bar.progress(min(frame_number / total_frames, 1.0))
                            status_text.write(f"Processing frame {frame_number}/{total_frames}")

                        cap.release()
                        writer.release()
                        progress_bar.progress(1.0)
                        status_text.success("✅ Fire & Smoke video processing completed!")

                        st.markdown("---")
                        st.subheader("🎯 Processed Video")
                        st.video(output_video.name)

                        st.subheader("📊 Fire & Smoke Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔥 Fire", max_fire)
                        with col2:
                            st.metric("💨 Smoke", max_smoke)
                        with col3:
                            st.metric("🎯 Total", max_fire + max_smoke)

                        st.info(
                            "Counts show the maximum number of Fire and Smoke detections "
                            "visible simultaneously in a single frame."
                        )


# ============================================================
# EMERGENCY VEHICLE DETECTION
# ============================================================

elif page == "🚑 Emergency Vehicle":

    st.title("🚑 Emergency Vehicle Detection")
    st.write(
        "Detect Ambulance, Fire Truck, and Police Vehicle using the trained YOLO11 model."
    )

    if not EMERGENCY_MODEL_PATH.exists():
        st.error(
            f"❌ Emergency Vehicle model not found: {EMERGENCY_MODEL_PATH}"
        )
    else:
        emergency_model = load_emergency_model()

        emergency_confidence = st.slider(
            "Detection Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            key="emergency_confidence"
        )

        input_type = st.radio(
            "Choose input",
            ["🖼️ Image", "🎥 Video"],
            horizontal=True,
            key="emergency_input_type"
        )

        if input_type == "🖼️ Image":

            uploaded_emergency_image = st.file_uploader(
                "📤 Upload Emergency Vehicle Image",
                type=["jpg", "jpeg", "png"],
                key="emergency_image_upload"
            )

            if uploaded_emergency_image is not None:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_emergency_image.name).suffix or ".jpg"
                ) as temp_file:
                    temp_file.write(uploaded_emergency_image.read())
                    emergency_image_path = temp_file.name

                st.subheader("📷 Original Image")
                st.image(emergency_image_path, use_container_width=True)

                if st.button(
                    "🔍 Detect Emergency Vehicles",
                    key="emergency_image_detection"
                ):
                    with st.spinner("🚑 Analyzing emergency vehicles in overlapping tiles..."):
                        original_emergency_image = cv2.imread(
                            emergency_image_path
                        )

                        emergency_annotated, emergency_counts, _ = (
                            detect_emergency_with_tiles(
                                original_emergency_image,
                                emergency_model,
                                confidence=emergency_confidence,
                                tile_size=640,
                                overlap=0.25
                            )
                        )

                    st.subheader("🎯 Emergency Vehicle Detection Result")
                    st.image(
                        emergency_annotated,
                        channels="BGR",
                        use_container_width=True
                    )

                    st.markdown("---")
                    st.subheader("📊 Emergency Vehicles Detected")

                    if not emergency_counts:
                        st.warning("No emergency vehicles detected.")
                    else:
                        st.success(
                            f"🚨 Total Emergency Vehicles: {sum(emergency_counts.values())}"
                        )
                        cols = st.columns(min(len(emergency_counts), 3))
                        for index, (name, count) in enumerate(emergency_counts.items()):
                            with cols[index % len(cols)]:
                                st.metric(name.title(), count)

                try:
                    os.remove(emergency_image_path)
                except OSError:
                    pass

        else:

            uploaded_emergency_video = st.file_uploader(
                "📤 Upload Emergency Vehicle Video",
                type=["mp4", "avi", "mov", "mkv"],
                key="emergency_video_upload"
            )

            if uploaded_emergency_video is not None:

                input_emergency_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                input_emergency_video.write(uploaded_emergency_video.read())
                input_emergency_video.close()

                st.subheader("🎥 Original Video")
                st.video(input_emergency_video.name)

                if st.button(
                    "🔍 Detect Emergency Vehicles in Video",
                    key="emergency_video_detection"
                ):

                    cap = cv2.VideoCapture(input_emergency_video.name)

                    if not cap.isOpened():
                        st.error("❌ Unable to open the video.")
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30

                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                        output_emergency_video = tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        )
                        output_emergency_video.close()

                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            output_emergency_video.name,
                            fourcc,
                            fps,
                            (width, height)
                        )

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        max_emergency_counts = {
                            "Ambulance": 0,
                            "Fire Truck": 0,
                            "Police Vehicle": 0
                        }

                        frame_number = 0

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_number += 1

                            results = emergency_model.predict(
                                source=frame,
                                conf=emergency_confidence,
                                imgsz=640,
                                verbose=False
                            )
                            result = results[0]

                            current_counts = Counter()
                            if result.boxes is not None:
                                for class_id in result.boxes.cls:
                                    class_id = int(class_id)
                                    class_name = emergency_model.names[class_id]
                                    current_counts[class_name] += 1

                            for class_name in max_emergency_counts:
                                max_emergency_counts[class_name] = max(
                                    max_emergency_counts[class_name],
                                    current_counts.get(class_name, 0)
                                )

                            annotated_frame = result.plot(
                                labels=True,
                                conf=True,
                                boxes=True
                            )

                            current_total = sum(current_counts.values())
                            cv2.putText(
                                annotated_frame,
                                f"Emergency Vehicles: {current_total}",
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 0),
                                3
                            )

                            writer.write(annotated_frame)

                            if total_frames > 0:
                                progress = frame_number / total_frames
                                progress_bar.progress(min(progress, 1.0))

                            status_text.write(
                                f"Processing frame {frame_number}/{total_frames}"
                            )

                        cap.release()
                        writer.release()
                        progress_bar.progress(1.0)
                        status_text.success("✅ Emergency vehicle video processing completed!")

                        st.markdown("---")
                        st.subheader("🎯 Processed Video")
                        st.video(output_emergency_video.name)

                        st.subheader("📊 Emergency Vehicle Statistics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("🚑 Ambulance", max_emergency_counts["Ambulance"])
                        with col2:
                            st.metric("🚒 Fire Truck", max_emergency_counts["Fire Truck"])
                        with col3:
                            st.metric("🚓 Police Vehicle", max_emergency_counts["Police Vehicle"])

                        st.info(
                            "Counts show the maximum number of each emergency vehicle "
                            "visible simultaneously in a single frame."
                        )


# ============================================================
# ALERTS
# ============================================================

elif page == "Alerts":

    st.title("🚨 Alerts")

    st.info(
        "Intelligent alerts will be developed during "
        "the final stage of the project."
    )

    st.write(
        """
        Planned final features:

        🔥 Fire / Smoke Detection

        🚑 Emergency Vehicle Detection

        🚦 Traffic Congestion

        🔄 Wrong-Way Detection

        ⚠️ Anomaly / Accident Detection

        🚧 Restricted-Zone Detection

        🧍 Crowd Intelligence

        🔮 Prediction

        🚨 Automatic Alerts
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI City Surveillance | YOLO11"
)