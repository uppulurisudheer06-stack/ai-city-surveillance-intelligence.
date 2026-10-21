# AI City Surveillance Intelligence Platform

## 1. Introduction
The AI City Surveillance Intelligence Platform is a computer vision project that explores the automated analysis of images and videos. It uses deep learning-based object detection to identify selected events, including emergency vehicles, fire, and smoke.

## 2. Problem Statement
Manually monitoring large amounts of surveillance footage can be time-consuming. This project aims to help users analyse visual information using object detection models and an interactive application.

## 3. Objectives
- Detect objects in supported images and videos.
- Identify emergency vehicles using the available detection model.
- Detect fire and smoke using a trained model.
- Display detection results through a user-friendly interface.
- Evaluate the models using suitable test data and performance metrics.

## 4. Technologies Used
- **Python:** Main programming language.
- **YOLO:** Object detection framework.
- **OpenCV:** Image and video processing.
- **Streamlit:** Interactive application interface.
- **GitHub:** Source code management and project documentation.

## 5. Methodology
1. **Dataset collection:** Select suitable image and video datasets for the detection tasks.
2. **Preprocessing:** Prepare images and annotations in the format required by the models.
3. **Model preparation:** Load the relevant trained YOLO model weights.
4. **Testing:** Test the models using suitable images and videos.
5. **Application integration:** Connect the models to the Streamlit application.
6. **Evaluation:** Record detection examples and measure suitable performance metrics.

## 6. Project Modules
- General object detection
- Emergency vehicle detection
- Fire and smoke detection
- Image and video analysis
- Interactive results display

## 7. Project Files
- `app.py` — Main Streamlit application.
- `index.html` — Project information webpage.
- `requirements.txt` — Python package requirements.
- `check_labels.py` — Supporting label-checking script.
- `fix_labels.py` — Supporting label-correction script.

## 8. How to Run the Application
1. Install Python on your computer.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Ensure that the required model weights and datasets are available at the paths expected by the application.
6. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local URL displayed in the terminal.

## 9. Testing and Evaluation
Test the application with suitable images and videos. Record the number of test samples, successful detections, missed detections, and inference time. Where ground-truth annotations are available, evaluate precision, recall, mAP@0.5, and mAP@0.5:0.95.

Actual numerical results and screenshots should be added after testing. No performance scores are claimed in this document.

## 10. Limitations
- Detection performance depends on the quality and variety of training data.
- Lighting, image quality, object size, and environmental conditions may affect predictions.
- The application requires the correct model weights and Python dependencies.

## 11. Future Work
- Expand testing with more diverse datasets.
- Improve detection performance under different lighting conditions.
- Add more evaluation results and visual examples.
- Explore additional surveillance-related detection tasks.

## 12. Team Members
- **Sudheer** — CB.AI.U4AID25165
- **Venkat** — CB.AI.U425159
- **Abishek** — CB.AI.U4AID25101
- **Sanmukh Sai** — CB.AI.U4AID25161

## 13. References
Add the actual research papers, dataset sources, and official documentation used during project development.

## 14. Project Webpage
The project webpage is included in the repository as `index.html`. A public webpage link will be added after the site is published using GitHub Pages.
