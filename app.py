import streamlit as st
from logic import HomeInspector
import os
from pathlib import Path
import tempfile
import json
from io import BytesIO
import base64
from PIL import Image
import pandas as pd
import shutil
from dotenv import load_dotenv
load_dotenv()


# Page config
st.set_page_config(
    page_title="Home Inspection AI",
    page_icon="🏠",
    layout="wide"
)

def create_word_download_link(report_data):
    """Generate a download link for the Word report"""
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "inspection_report.docx")
    
    # Generate the Word report
    inspector = st.session_state.inspector
    inspector.generate_word_report(report_data, output_path)
    
    # Read the file and create download link
    with open(output_path, "rb") as f:
        bytes_data = f.read()
    
    # Create download button
    return bytes_data

# Sidebar for API key input
try:
    inspector = HomeInspector()
    st.session_state.inspector = inspector
    st.session_state.processed = False
    st.success("To build an AI-powered Home Inspection System that analyzes user-uploaded images or videos of a property to automatically generate a detailed inspection report")
except Exception as e:
    st.error(f"Error initializing inspector: {str(e)}")
    st.stop()    

# Main app
st.title("🏠 AI Home Inspection System")

st.markdown("Upload an image or a video of your home for a detailed inspection report")

if 'inspector' not in st.session_state:
    st.warning("Please initialize the inspector in the sidebar first")
    st.stop()

inspector = st.session_state.inspector

# Choose Image or Video
# --- Step 2: Upload Media ---

st.title("Step 2: Upload Media")
st.markdown("Upload an image or a video of your home for a detailed inspection report")
choice = st.radio("Choose media type:", ("Image", "Video"))

# File uploader
if choice == "Image":
    uploaded_files = st.file_uploader("Upload images of your home", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
else:
    uploaded_files = st.file_uploader("Upload a video of your home", type=["mp4", "mov", "avi"])

if uploaded_files and not st.session_state.get("processed", False):
    with st.spinner("Processing media..."):
        try:
            temp_dir = tempfile.mkdtemp()

            if choice == "Image":
                # Handling multiple image upload
                image_paths = []
                for uploaded_file in uploaded_files:
                    image_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_paths.append(image_path)

                inspector.upload_user_media(image_paths)

                # Display uploaded images
                cols = st.columns(min(5, len(uploaded_files)))
                for i, uploaded_file in enumerate(uploaded_files[:25]):
                    try:
                        image = Image.open(uploaded_file)
                        cols[i].image(image, caption=uploaded_file.name)
                    except Exception as e:
                        cols[i].warning(f"Could not preview {uploaded_file.name}: {str(e)}")

                st.session_state.processed = True

            elif choice == "Video":
                # Handling single video upload
                media_path = os.path.join(temp_dir, uploaded_files.name)
                with open(media_path, "wb") as f:
                    f.write(uploaded_files.getbuffer())

    # Process video to extract frames
                frame_paths = inspector.process_video(media_path)

    # Create destination directory
                save_dir = os.path.join("Datasets", "user_data-copy")
                os.makedirs(save_dir, exist_ok=True)

                saved_frame_paths = {}
                for frame_name, frame_path in frame_paths.items():
                    frame_file_name = frame_name + ".jpg"  # Ensure consistent filename
                    new_path = os.path.join(save_dir, frame_file_name)
                    shutil.copy(frame_path, new_path)
                    saved_frame_paths[frame_name] = new_path

    # Update frame_paths to point to saved frames
                frame_paths = saved_frame_paths

                st.session_state.frame_paths = frame_paths
                st.session_state.video_path = media_path

        # Upload both the video and extracted frames
                inspector.upload_user_media([media_path] + list(frame_paths.values()))

                st.session_state.processed = True
                st.session_state.video_processed = True

                st.success(f"Video processed successfully! Extracted {len(frame_paths)} frames.")

        # Show first few frames
                cols = st.columns(5)
                for i, (name, path) in enumerate(list(frame_paths.items())[:25]):
                    try:
                        cols[i].image(path, caption=f"Frame {name}")
                    except Exception as e:
                        cols[i].warning(f"Could not preview frame {name}: {str(e)}")

        except Exception as e:
            st.error(f"Error processing media: {str(e)}")



# Generate Report
if st.session_state.get("processed", False):
    if st.button("Generate Inspection Report"):
        with st.spinner("Generating report (this may take a few minutes)..."):
            try:
                report = inspector.generate_report()
                
                # Ensure media references are properly set for video frames
                if st.session_state.get("video_processed", False) and 'frame_paths' in st.session_state:
                    frame_paths = st.session_state.frame_paths
                    for finding in report['detailedInspection']:
                        if finding.get('mediaReference'):
                            frame_name = finding['mediaReference']
                            if frame_name in frame_paths:
                                finding['mediaReference'] = os.path.basename(frame_paths[frame_name])
                
                st.session_state.report = report
                
                with open("inspection_report.json", "w") as f:
                    json.dump(report, f, indent=4)
                
                st.session_state.report_ready = True
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

# Display report if available
if st.session_state.get("report_ready", False):
    report = st.session_state.report
    
    st.header("Inspection Report")
    
    # Executive Summary
    with st.expander("Executive Summary", expanded=True):
        st.subheader("Overall Condition")
        st.write(report['executiveSummary']['overallCondition'])
        
        st.subheader("Critical Issues")
        for issue in report['executiveSummary']['criticalIssues']:
            st.error(f"⚠️ {issue}")
            
        st.subheader("Recommended Actions")
        for action in report['executiveSummary']['recommendedActions']:
            st.info(f"🔧 {action}")
    
    # Detailed Inspection
    st.header("Detailed Inspection Findings")
    for finding in report['detailedInspection']:
        with st.expander(f"{finding['area']} - {finding['condition']}", expanded=False):
            cols = st.columns([1, 3])
            
            # Show image if available
            if finding.get('mediaReference'):
                media_ref = finding['mediaReference']
                media_path = os.path.join("Datasets", "user_data-copy", media_ref)
                if os.path.exists(media_path):
                    cols[0].image(media_path, caption=f"Media reference: {media_ref}")
                else:
                    cols[0].warning(f"Media not found: {media_ref}")
            
            # Show details
            with cols[1]:
                st.markdown(f"**Compliance Status:** `{finding['complianceStatus']}`")
                
                if finding.get('issuesFound'):
                    st.markdown("**Issues Found:**")
                    for issue in finding['issuesFound']:
                        st.markdown(f"- {issue}")
                
                if finding.get('referenceDoc') and finding.get('referenceSection'):
                    st.markdown(f"**Standard Reference:** {finding['referenceDoc']} - {finding['referenceSection']}")
                
                if finding.get('recommendation'):
                    st.markdown(f"**Recommendation:** {finding['recommendation']}")
    
    # Maintenance Notes
    with st.expander("Maintenance Schedule", expanded=False):
        for schedule in report['maintenanceNotes']['maintenanceSchedule']:
            st.subheader(f"{schedule['frequency']} Tasks")
            for task in schedule['tasks']:
                st.markdown(f"- {task}")
        
        if report['maintenanceNotes'].get('costConsiderations'):
            st.subheader("Cost Considerations")
            for cost in report['maintenanceNotes']['costConsiderations']:
                st.markdown(f"- {cost}")
    
    # Download buttons
    st.subheader("Download Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(report, indent=4),
            file_name="home_inspection_report.json",
            mime="application/json"
        )
    
    with col2:
        word_bytes = create_word_download_link(report)
        st.download_button(
            label="Download Word Report",
            data=word_bytes,
            file_name="home_inspection_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# BI Report Section
if st.session_state.get("report_ready", False):
    if st.button("View Report Dashboard"):
        st.session_state.show_bi_report = True

if st.session_state.get("show_bi_report", False):
    st.markdown("---")
    st.title("📊 Building Inspection BI Report Dashboard")

    response_json = st.session_state.report

    # Summary
    st.header("Summary")
    critical_issues = sum(1 for item in response_json.get('detailedInspection', []) if item.get('complianceStatus') == 'Non-compliant')
    high_priority_tasks = len([item for item in response_json.get('detailedInspection', []) if item.get('complianceStatus') == 'Non-compliant'])
    st.metric("Critical Issues", critical_issues)
    st.metric("High Priority Tasks", high_priority_tasks)

    # Helper functions
    def parse_inspection_table(response_json):
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

    # Inspection Results with Images
    st.header("Inspection Results")

    for inspection in response_json.get('detailedInspection', []):
        st.subheader(f"{inspection.get('area', 'N/A')} - {inspection.get('condition', 'N/A')}")

        cols = st.columns([1, 4])  # Left column for image, right for details

        # Display image on the left
        media_ref = inspection.get('mediaReference')
        if media_ref:
            media_path = os.path.join("Datasets", "user_data-copy", media_ref)
            if os.path.exists(media_path):
                try:
                    if media_ref.lower().endswith(('.jpg', '.jpeg', '.png')):
                        cols[0].image(media_path, caption=media_ref, width=200)
                    elif media_ref.lower().endswith(('.mp4', '.mov', '.avi')):
                        cols[0].video(media_path)
                except Exception as e:
                    cols[0].warning(f"Could not display media: {str(e)}")
            else:
                cols[0].warning(f"Media not found at: {media_path}")
        else:
            cols[0].write("No media reference.")

        # Show inspection details on the right
        with cols[1]:
            st.markdown(f"**Compliance Status:** `{inspection.get('complianceStatus', 'N/A')}`")
            if inspection.get('issuesFound'):
                st.markdown("**Issues Found:**")
                for issue in inspection['issuesFound']:
                    st.markdown(f"- {issue}")
            if inspection.get('recommendation'):
                st.markdown(f"**Recommendation:** {inspection['recommendation']}")
            if inspection.get('referenceDoc') and inspection.get('referenceSection'):
                st.markdown(f"**Standard Reference:** {inspection['referenceDoc']} - {inspection['referenceSection']}")

    # Maintenance
    st.header("Maintenance Schedule")
    maintenance_df = parse_maintenance_schedule(response_json)
    if not maintenance_df.empty:
        st.dataframe(maintenance_df)
    else:
        st.write("No maintenance schedule available.")

    # Chat (placeholder)
    st.header("Ask Questions About the Inspection")
    user_input = st.text_input("Ask a question about the inspection:")
    if st.button("Send"):
        if user_input:
            st.write("Response:")
            st.write("This is where the response from the model would be displayed.")
        else:
            st.warning("Please enter a question.")
