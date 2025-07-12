import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to log pose performance
def log_pose_data(pose_name, duration, calories):
    log_file = "pose_history.csv"
    data = {
        "DateTime": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Pose": [pose_name],
        "TimeHeldSeconds": [round(duration, 2)],
        "CaloriesBurned": [round(calories, 2)]
    }
    df = pd.DataFrame(data)

    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, mode='w', header=True, index=False)

# Calorie multiplier function based on age and gender
def get_calorie_multiplier(age, gender):
    if gender == "Male":
        return 1.1 if age < 30 else 1.0
    elif gender == "Female":
        return 1.0 if age < 30 else 0.9
    return 1.0

# Function to check pose correctness
def is_pose_correct(results, selected_pose):
    if not results.pose_landmarks:
        return False

    landmarks = results.pose_landmarks.landmark

    if selected_pose == "Warrior 2 Pose":
        # Existing Warrior 2 Pose logic
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        wrist_distance_from_shoulder = np.abs(left_wrist[1] - left_shoulder[1]) < 0.1 and \
                                       np.abs(right_wrist[1] - right_shoulder[1]) < 0.1

        return (160 <= left_arm_angle <= 180 and 160 <= right_arm_angle <= 180) and wrist_distance_from_shoulder

    elif selected_pose == "Warrior 1 Pose":
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        front_knee_angle = calculate_angle([left_ankle.x, left_ankle.y], [left_knee.x, left_knee.y], [left_hip.x, left_hip.y])
        back_leg_angle = calculate_angle([right_ankle.x, right_ankle.y], [right_knee.x, right_knee.y], [right_hip.x, right_hip.y])
        hands_up = (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)

        return (80 <= front_knee_angle <= 110 and back_leg_angle > 160 and hands_up)


    elif selected_pose == "Triangle Pose":
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Calculate vertical arm line
        left_arm_angle = calculate_angle(left_wrist, left_shoulder, left_hip)
        right_arm_angle = calculate_angle(right_wrist, right_shoulder, right_hip)

        # Calculate shoulder-hip alignment to avoid torso bending forward
        torso_alignment = np.abs(left_shoulder[2] - left_hip[2]) if len(left_shoulder) > 2 else 0

        # Check if one arm is straight above and the other is down (forming a line), and torso is aligned
        if (170 <= left_arm_angle <= 190 and 170 <= right_arm_angle <= 190):
            return True
        elif (170 <= right_arm_angle <= 190 and 170 <= left_arm_angle <= 190):
            return True
        return False


    elif selected_pose == "Side Stretch":
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Angle between arm, shoulder, and hip
        left_side_angle = calculate_angle(left_wrist, left_shoulder, left_hip)
        right_side_angle = calculate_angle(right_wrist, right_shoulder, right_hip)

        # Check if either side shows proper stretch
        return (160 <= left_side_angle <= 200 or 160 <= right_side_angle <= 200)


    elif selected_pose == "Tree Pose":
        # Existing Tree Pose logic
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        distance_right_leg = np.sqrt((right_ankle.x - left_knee.x) ** 2 + (right_ankle.y - left_knee.y) ** 2)
        distance_left_leg = np.sqrt((left_ankle.x - right_knee.x) ** 2 + (left_ankle.y - right_knee.y) ** 2)

        if distance_right_leg < 0.1 or distance_left_leg < 0.1:
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            hands_up = left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y
            return hands_up

    elif selected_pose == "T Pose":
        # Existing T Pose logic
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        wrist_level = np.abs(left_wrist[1] - left_shoulder[1]) < 0.1 and \
                      np.abs(right_wrist[1] - right_shoulder[1]) < 0.1

        return (170 <= left_angle <= 190 and 170 <= right_angle <= 190) and wrist_level

    return False

# App Title
st.title("AI Based Yoga Trainer")

# Sidebar Pose Selection
selected_pose = st.sidebar.selectbox("Select a Pose", ["Warrior 1 Pose", "Warrior 2 Pose", "Triangle Pose", "Side Stretch", "Tree Pose", "T Pose"])

# Sidebar User Info
st.sidebar.markdown("## 🧑‍💻 User Info")
age = st.sidebar.number_input("Enter your age", min_value=5, max_value=100, value=25)
gender = st.sidebar.selectbox("Select Gender", ["Male", "Female"])

# Display Image Guide for Selected Pose
pose_images = {
    "Warrior 1 Pose": "images/Warrior-1.jpg",
    "Warrior 2 Pose": "images/warrior_2_pose.png",
    "Triangle Pose": "images/triangle.jpg",
    "Side Stretch": "images/Side Stretch.jpg",
    "Tree Pose": "images/tree_pose.jpg",
    "T Pose": "images/t_pose.jpeg"
}

if selected_pose in pose_images:
    try:
        img = Image.open(pose_images[selected_pose])
        st.image(img, caption=f"{selected_pose} Guide", use_container_width= True)
    except Exception as e:
        st.warning(f"Image for {selected_pose} not found or cannot be loaded.")

# Define calories per second by pose (example values)
pose_calorie_rates = {
    "Warrior 1 Pose": 0.07,
    "Warrior 2 Pose": 0.06,
    "Triangle Pose": 0.065,
    "Side Stretch": 0.05,
    "Tree Pose": 0.045,
    "T Pose": 0.04
}

# Live Camera Section
st.write("### Live Camera Feed")
run = st.checkbox("Start Camera")
st.write("### Instructions:")
st.write("1. Select a pose from the dropdown menu.")
st.write("2. Click on 'Start Camera' to begin pose detection.")
st.write("3. Maintain the pose to see the timer.")
st.write("4. If the pose becomes incorrect, you'll see the maximum time held.")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access the camera.")
    else:
        frame_window = st.image([])
        pose_correct = False
        start_time = None
        max_pose_time = 0
        correct_pose_time = 0
        timer_placeholder = st.empty()
        total_calories = 0.0

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from the camera.")
                break

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if is_pose_correct(results, selected_pose):
                if not pose_correct:
                    start_time = time.time()
                    pose_correct = True
                    st.balloons()
                elapsed_time = time.time() - start_time
                correct_pose_time = elapsed_time
                multiplier = get_calorie_multiplier(age, gender)
                calorie_rate = pose_calorie_rates.get(selected_pose, 0.05)
                total_calories = correct_pose_time * calorie_rate * multiplier
                max_pose_time = max(max_pose_time, correct_pose_time)
                timer_placeholder.write(f"✅ Pose Correct! Time: {correct_pose_time:.2f} sec | Calories: {total_calories:.2f} cal")
            else:
                if pose_correct:
                    pose_correct = False
                    multiplier = get_calorie_multiplier(age, gender)
                    calorie_rate = pose_calorie_rates.get(selected_pose, 0.05)
                    total_calories = max_pose_time * calorie_rate * multiplier
                    timer_placeholder.write(f"❌ Pose Incorrect! Max Time Held: {max_pose_time:.2f} sec | Calories: {total_calories:.2f} cal")
                    log_pose_data(selected_pose, max_pose_time, total_calories)
                    correct_pose_time = 0
                    max_pose_time = 0

            frame_window.image(frame)

        cap.release()

# Pose History and Progress Tracker
st.markdown("## 📊 Pose History & Progress Tracker")
if st.button("Show My Progress"):
    log_file = "pose_history.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)

        st.dataframe(df)

        total = df["CaloriesBurned"].sum()
        st.success(f"🔥 Total Calories Burned: {total:.2f} cal")

        # ✅ Line Chart - Calories Burned Over Time
        st.write("### 📈 Calories Burned Over Time")
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df_sorted = df.sort_values("DateTime")
        fig_line, ax_line = plt.subplots()
        ax_line.plot(df_sorted["DateTime"], df_sorted["CaloriesBurned"].cumsum(), label="Calories Burned", marker="o", color="orange")
        ax_line.set_xlabel("DateTime")
        ax_line.set_ylabel("Calories Burned")
        ax_line.set_title("Cumulative Calories Burned Over Time")
        ax_line.grid(True)
        ax_line.legend()
        st.pyplot(fig_line)

        # ✅ Bar Chart - Pose-wise Calorie Burn by Date
        st.write("### 📊 Calories Burned per Pose per Day")
        df["Date"] = df["DateTime"].dt.date
        grouped = df.groupby(["Date", "Pose"])["CaloriesBurned"].sum().reset_index()
        pivot_df = grouped.pivot(index="Date", columns="Pose", values="CaloriesBurned").fillna(0)

        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Paired.colors  # Color map
        pivot_df.plot(kind='bar', stacked=True, ax=ax_bar, color=colors)

        ax_bar.set_title("Calories Burned per Pose per Day")
        ax_bar.set_xlabel("Date")
        ax_bar.set_ylabel("Calories Burned")
        ax_bar.legend(title="Pose", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_bar.grid(True)
        st.pyplot(fig_bar)

    else:
        st.warning("No progress data found yet. Do some yoga to start tracking!")

