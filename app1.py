import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import base64
from PIL import Image

# Load the JSON response
response_json_path = "home_inspection.json"
if os.path.exists(response_json_path):
    with open(response_json_path, "r") as file:
        response_json = json.load(file)
else:
    st.error("Response JSON file not found.")
    response_json = {}

# Helper functions
def get_image_base64(image_path):
    """Convert image file to base64 string."""
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None

def parse_inspection_table(response_json):
    """Extract inspection table data from JSON response."""
    inspection_data = []
    for inspection in response_json.get('detailedInspection', []):
        inspection_data.append({
            'Area': inspection.get('area', 'N/A'),
            'Media Reference': inspection.get('mediaReference', 'N/A'),
            'Condition': inspection.get('condition', 'N/A'),
            'Compliance Status': inspection.get('complianceStatus', 'N/A'),
            'Issues Found': ', '.join(inspection.get('issuesFound', [])),
            'Recommendation': inspection.get('recommendation', 'N/A')
        })
    return pd.DataFrame(inspection_data)

def parse_maintenance_schedule(response_json):
    """Extract maintenance schedule from JSON response."""
    schedule_items = []
    for inspection in response_json.get('detailedInspection', []):
        if inspection.get('complianceStatus') == 'Non-compliant':
            schedule_items.append({
                'Task': inspection.get('recommendation', 'N/A'),
                'Frequency': 'Immediate',
                'Priority': 'High',
                'Status': 'Pending'
            })
    return pd.DataFrame(schedule_items)

# Streamlit app layout
st.title("Building Inspection Report Dashboard")

# Summary Section
st.header("Summary")
critical_issues = sum(1 for item in response_json.get('detailedInspection', []) if item.get('complianceStatus') == 'Non-compliant')
high_priority_tasks = len([item for item in response_json.get('detailedInspection', []) if item.get('complianceStatus') == 'Non-compliant'])
st.metric("Critical Issues", critical_issues)
st.metric("High Priority Tasks", high_priority_tasks)

# Inspection Results Section
st.header("Inspection Results")
inspection_data = parse_inspection_table(response_json)
if not inspection_data.empty:
    st.dataframe(inspection_data)
else:
    st.write("No inspection data available.")

# Maintenance Schedule Section
st.header("Maintenance Schedule")
maintenance_schedule = parse_maintenance_schedule(response_json)
if not maintenance_schedule.empty:
    st.dataframe(maintenance_schedule)
else:
    st.write("No maintenance schedule available.")

# Media Viewer Section
st.header("Media Viewer")
media_files = response_json.get('detailedInspection', [])
for media in media_files:
    media_ref = media.get('mediaReference', 'N/A')
    st.subheader(f"Media: {media_ref}")
    if media_ref.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join("Datasets/user_data", media_ref)
        if os.path.exists(image_path):
            st.image(image_path, caption=media_ref)
        else:
            st.write(f"Image not found: {media_ref}")
    elif media_ref.endswith('.mp4'):
        video_path = os.path.join("Datasets/user_data", media_ref)
        if os.path.exists(video_path):
            st.video(video_path)
        else:
            st.write(f"Video not found: {media_ref}")

# Chat Section

st.header("Ask Questions About the Inspection")
user_input = st.text_input("Ask a question about the inspection:")
if st.button("Send"):
    if user_input:
        st.write("Response:")
        st.write("This is where the response from the model would be displayed.")
    else:
        st.warning("Please enter a question.")
