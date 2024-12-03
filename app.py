import streamlit as st
import subprocess

# Title and introduction
st.title("Workout Tracker")
st.markdown("""
Welcome to the **Workout Tracker App**! 
Select your desired workout below, and the app will guide you through the exercise with real-time feedback.
""")

# Workout options
st.header("Choose Your Workout")
workout_option = st.selectbox(
    "Available Workouts:",
    ["Bicep Curl", "Lateral Raise", "Shoulder Press"]
)

# Button to start the workout
if st.button("Start Workout"):
    st.write(f"Starting {workout_option}...")

    # Map the workout to the corresponding script
    workout_scripts = {
        "Bicep Curl": "bicep_curl.py",
        "Lateral Raise": "lateral_raise.py",
        "Shoulder Press": "shoulder_press.py",
    }
    
    selected_script = workout_scripts.get(workout_option)
    
    # Run the corresponding script
    try:
        subprocess.run(["python", selected_script], check=True)
        st.success(f"{workout_option} workout completed! Check the feedback on your terminal.")
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while running {workout_option}. Please try again.")
    except FileNotFoundError:
        st.error(f"Workout script {selected_script} not found! Ensure the file exists in the same directory.")

# Footer
st.markdown("""
---
**Note**: Close the workout window or press "q" in the camera feed to stop the workout.
""")
