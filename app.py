import os
import json
import time  
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import datetime
#from kaggle_secrets import UserSecretsClient
import requests
import shutil
from pathlib import Path
from google.generativeai import caching
import pickle
import dotenv
import time
import json
import google.generativeai as genai

api_key = "AIzaSyCEMGQqrRS35K1BKieVrtA6ZmP8TAyFUrE"
genai.configure(api_key=api_key)
from pathlib import Path

# Australian building standards
docs_path = Path(r".\Datasets\building_standards")

# Professionally generated building report used to teach the LLM
examples_path = Path(r".\Datasets\examples")

# User uploaded photos of their house
user_path = Path(r".\Datasets\user_data")

# User uploaded video of their house (concatenated into a single video)
video_file_name = r"Datasets/user_data/home_inspection.mp4"
# We upload files and store in a dictionairy by folder and subfolder
document_dict = {
    'building_standards': {},
    'examples': {
        'example1': {},
        'example2': {}
    },
    'user_data': {}
}

# Supported file extensions (add more if needed)
SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png'}

# Load building standards documents
for file_path in docs_path.rglob('*'):
    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        try:
            uploaded_file = genai.upload_file(str(file_path))
            document_dict['building_standards'][file_path.name] = uploaded_file
            print(f"Loaded standard: {file_path.name}")
        except Exception as e:
            print(f"Error loading standard {file_path.name}: {str(e)}")

# Load user media
for file_path in user_path.rglob('*'):
    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        try:
            uploaded_file = genai.upload_file(str(file_path))
            document_dict['user_data'][file_path.name] = uploaded_file
            print(f"Loaded user media: {file_path.name}")
        except Exception as e:
            print(f"Error loading user media {file_path.name}: {str(e)}")

# Load example images, handling subfolders
for file_path in examples_path.rglob('*'):
    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        try:
            # Get the parent folder name (example1 or example2)
            subfolder = file_path.parent.name
            if subfolder in ['example1', 'example2']:
                uploaded_file = genai.upload_file(str(file_path))
                document_dict['examples'][subfolder][file_path.name] = uploaded_file
                print(f"Loaded example image from {subfolder}: {file_path.name}")
        except Exception as e:
            print(f"Error loading example image {file_path.name}: {str(e)}")

# Handle video file separately
print(f"Uploading video file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")

while video_file.state.name == "PROCESSING":
    print('Waiting for video to be processed.')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
print(f'Video processing complete: ' + video_file.uri)

# Add video to user_data dictionary
document_dict['user_data'][Path(video_file_name).name] = video_file
# Load example JSON files (for in-context learning)
example_jsons = {}
with open(".\\Datasets\\examples\\example1\\example1.json", 'r') as f:
    example_jsons['example1'] = json.load(f)
with open(".\\Datasets\\examples\\example2\\example2.json", 'r') as f:
    example_jsons['example2'] = json.load(f)
# Initialise the model and cache inputs (to save money)
cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-002', # for best results use -pro-002, however -flash-002 can be used on the free tier
    display_name='home_inspection_cache', # used to identify the cache
    system_instruction=(
        'You are an expert at analysing residential building and producing detailed inspection reports.'
        'Your job is to analyse the user provided media and produce a detailed inspection report based on the reference standards you have access to.'
    ),
    contents=[doc for doc in document_dict['building_standards'].values()], # if you have 503 errors you may need to reduce the amount of cached content
    ttl=datetime.timedelta(minutes=60),
)

generation_config = {
  "temperature": 0.1,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

# Construct a GenerativeModel which uses the created cache.
model = genai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=generation_config)
prompt = """
You have been supplied with a set of building standards and manufacturer specifications to evaluate the photos and videos against.
Please be specific about any violations of building codes or manufacturer specifications found in the documentation.

Analyze the uploaded photos and videos of the building and generate a detailed inspection report in JSON format.
Be exhaustive in your inspection and cover all aspects of the building shown in the media.

The response should be a valid JSON object with the following structure:

{
  "detailedInspection": [
    {
      "area": "string",
      "mediaReference": "string",
      "timestamp": "string",
      "condition": "string",
      "complianceStatus": "string",
      "issuesFound": ["string"],
      "referenceDoc": "string",
      "referenceSection": "string",
      "recommendation": "string"
    }
  ],
  "executiveSummary": {
    "overallCondition": "string",
    "criticalIssues": ["string"],
    "recommendedActions": ["string"]
  },
  "maintenanceNotes": {
    "recurringIssues": ["string"],
    "preventiveRecommendations": ["string"],
    "maintenanceSchedule": [
      {
        "frequency": "string",
        "tasks": ["string"]
      }
    ],
    "costConsiderations": ["string"]
  }
}

Ensure the response is a valid JSON object that can be parsed.
"""

content = []

# Add prompt
content.append({
    'text': prompt
})

# Add example header
content.append({
    'text': 'Here are some examples of analysed building reports:'
})

# Add example 1
content.append({
    'text': 'Example 1 Media and report (purely for reference):'
})

# Add example 1 media with document names
for name, doc in document_dict['examples']['example1'].items():
    content.append({
        'text': f"Example 1 Document: {name}"
    })
    content.append(doc)

# Add example 1 JSON
content.append({
    'text': json.dumps(example_jsons['example1'])
})

# Add example 2
content.append({
    'text': 'Example 2 Media and report (purely for reference):'
})

# Add example 2 media with document names
for name, doc in document_dict['examples']['example2'].items():
    content.append({
        'text': f"Example 2 Document: {name}"
    })
    content.append(doc)

# Add example 2 JSON
content.append({
    'text': json.dumps(example_jsons['example2'])
})

# Add user media header
content.append({
    'text': 'Now analyse the user provided media and provide a detailed inspection report. Analyse only the user provided images and video. Do not analyse either example provided earlier. You should analyse the entire video file (home_inspection.mp4) and consider approximately every 5 seconds as a unique timepoint to analyse as well as each image provided:'
})

content.append({
    'text': 'User provided media:'
})

# Add user media with document names
for name, doc in document_dict['user_data'].items():
    content.append({
        'text': f"User Document: {name}"
    })
    content.append(doc)

# Start chat with properly formatted content
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": content
        }
    ]
)

# Get the response
response = chat_session.send_message("Please generate a detailed building report. Please provide a detailed answer with elaboration on the report and reference material.")

# Print the response and token usage
print(response.text)
print("\nToken Usage:")
print(response.usage_metadata)

response_json = json.loads(response.text)
print(json.dumps(response_json, 
                indent=2,         
                sort_keys=True,    
                ensure_ascii=False
))
import json

# Ensure response_json is properly defined before saving
if 'response_json' in globals():  # Check if response_json exists
    with open("home_inspection.json", "w") as file:
        json.dump(response_json, file, indent=4)  # Pretty-print with indent=4
    print("File saved successfully as home_inspection.json")
else:
    print("Error: response_json is not defined")
with open(".\home_inspection.json", "w") as file:
    json.dump(response_json, file, indent=4)
print("File saved successfully on Desktop!")
import cv2

def timestamp_to_seconds(timestamp: str) -> float:
    """Convert timestamp string (MM:SS or HH:MM:SS) to seconds"""
    try:
        time_parts = timestamp.split(':')
        if len(time_parts) == 2:  # MM:SS
            minutes, seconds = time_parts
            return int(minutes) * 60 + int(seconds)
        elif len(time_parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = time_parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except:
        return 0


def extract_media_references(response_json: dict, document_dict: dict) -> dict:
    """
    Extract media references from JSON response and map them to actual files/timestamps
    Returns a dict mapping media references to file paths/timestamps
    """
    # Get all user media files from document_dict
    user_files = document_dict['user_data']
    media_refs = {}
    
    # Find the video file (assuming there's only one)
    video_file = next((file for filename, file in user_files.items() 
                      if file.mime_type.startswith('video/')), None)
    
    # Process detailed inspection entries
    for inspection in response_json['detailedInspection']:
        media = inspection.get('mediaReference', '')
        timestamp = inspection.get('timestamp', '')
        
        # Handle images
        if media.lower().endswith(('.jpg', '.jpeg', '.png')):
            if media in user_files:
                file = user_files[media]
                if file.mime_type.startswith('image/'):
                    media_refs[media] = {
                        'type': 'image',
                        'file': file,
                        'original_filename': media,
                        'timestamp': 'N/A'
                    }
        
        # Handle video references - Updated to handle "[HH:MM:SS]" format
        elif 'home_inspection.mp4' in media and timestamp:
            # Clean up timestamp if needed
            clean_timestamp = timestamp.strip('[]')
            if video_file:
                media_refs[f"home_inspection.mp4_{clean_timestamp}"] = {
                    'type': 'video',
                    'file': video_file,
                    'original_filename': Path(video_file_name).name,
                    'start_time': clean_timestamp,
                    'end_time': clean_timestamp
                }
    
    return media_refs
    
    
def extract_video_frames(media_references: dict, document_dict: dict, output_dir: str = 'extracted_frames') -> dict:
    """
    Extract frames from video at specified timestamps
    Returns dict mapping original references to frame file paths
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    frame_paths = {}
    
    # Group references by video file
    video_timestamps = []
    for ref, data in media_references.items():
        if data['type'] == 'video':
            timestamp = data['start_time']
            original_filename = data['original_filename']
            video_timestamps.append((ref, timestamp, original_filename))
    
    if not video_timestamps:
        print("No video timestamps found in media references")
        return frame_paths
    
    # Process each video file and its timestamps
    for ref, timestamp, video_filename in video_timestamps:
        
        # Open local video file
        cap = cv2.VideoCapture(str(video_file_name))
        if not cap.isOpened():
            print(f"Error opening video file: {video_file_name}")
            continue
        
        # Convert timestamp to frame position
        seconds = timestamp_to_seconds(timestamp)
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            # Generate output filename
            frame_filename = f"{ref.replace(':', '_').replace('.', '_')}.jpg"
            frame_path = str(Path(output_dir) / frame_filename)
            
            # Save frame as JPEG
            cv2.imwrite(frame_path, frame)
            frame_paths[ref] = frame_path
            print(f"Extracted frame at {timestamp} to {frame_path}")
        else:
            print(f"Failed to extract frame at {timestamp}")
        
        cap.release()
    
    return frame_paths

# Example usage:
media_references = extract_media_references(response_json, document_dict)
frame_paths = extract_video_frames(media_references, document_dict)
print("\nExtracted frames:", frame_paths)




from dash import html, dcc, dash_table, no_update
import dash_loading_spinners as dls 
from dash import Dash
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
import re
from dash.dependencies import Input, Output, State
import base64
from PIL import Image
import io
import os
base_dir = r".\Datasets"
print("Directory Exists:", os.path.exists(base_dir))
print("Files in Directory:", os.listdir(base_dir) if os.path.exists(base_dir) else "Path Not Found")
image_path = os.path.join(base_dir, "user_data")  
def get_image_base64(image_path):
    """Convert image file to base64 string"""
    if not os.path.exists(image_path):
        print(f"Error: File not found {image_path}")
        return None
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
def get_media_path(media_ref: str) -> str:
    """Get the full path for media reference"""
    print(f"Getting media path for {media_ref}") 
    if ',' in media_ref:
        media_ref = media_ref.split(',')[0].strip()
    if 'home_inspection.mp4' in media_ref:
        match = re.search(r'home_inspection.mp4 (\d{1,2}:\d{2})', media_ref)
        if match:
            timestamp = match.group(1).replace(':', '_')  
            return os.path.join(base_dir, 'extracted_frames', f'home_inspection_mp4_{timestamp}.jpg')
        return os.path.join(base_dir, 'user_data', media_ref)
    return os.path.join(base_dir, 'user_data', media_ref)
app = Dash(__name__)
def parse_inspection_table(response_json: dict) -> List[Dict]:
    """Extract inspection table data from JSON response"""
    inspection_data = []
    for inspection in response_json['detailedInspection']:
        details = []
        media_ref = inspection['mediaReference']
        if media_ref == 'home_inspection.mp4' and inspection.get('timestamp'):
            media_ref = f"home_inspection.mp4_{inspection['timestamp']}"
        if inspection.get('condition'):
            has_issues = (inspection.get('referenceDoc') or 
                        inspection.get('referenceSection') or 
                        inspection.get('issuesFound') or 
                        inspection.get('recommendation'))
            condition_text = inspection['condition']
            if not has_issues:
                condition_text += " - Good"
            details.append(f"<strong>Condition:</strong> {condition_text}")
        if inspection.get('referenceDoc') or inspection.get('referenceSection'):
            code_reference = f"{inspection.get('referenceDoc', 'N/A')} - {inspection.get('referenceSection', 'N/A')}"
            details.append(f"<strong>Code Reference:</strong> {code_reference}") 
        if inspection.get('issuesFound'):
            issues = '<br>'.join([f"• {issue}" for issue in inspection['issuesFound']])
            details.append(f"<strong>Issues Found:</strong><br>{issues}")    
        if inspection.get('recommendation'):
            details.append(f"<strong>Recommendation:</strong> {inspection['recommendation']}")    
        details_html = '<br><br>'.join(details)
        inspection_data.append({
            'Area': inspection['area'],
            'Media': media_ref, 
            'Details': details_html,
            'Priority': 'High' if inspection['complianceStatus'] == 'Non-compliant' else 'Medium' if 'Potentially' in inspection['complianceStatus'] else 'Low'
        })
    return inspection_data
def parse_maintenance_schedule(response_json: dict) -> List[Dict]:
    """Extract and generate maintenance schedule from inspection findings"""
    schedule_items = []
    for inspection in response_json['detailedInspection']:
        if inspection['complianceStatus'] == 'Non-compliant':
            recommendation = inspection.get('recommendation', '')
            issues = inspection.get('issuesFound', [])
            frequency = 'Immediate' if any(word in ' '.join(issues).lower() 
                                         for word in ['immediate', 'critical', 'urgent', 'termite', 'pest']) \
                       else 'Quarterly'
            schedule_items.append({
                'Task': recommendation,
                'Frequency': frequency,
                'Priority': 'High' if frequency == 'Immediate' else 'Medium',
                'Status': 'Pending'
            })
    standard_tasks = [
        {
            'Task': 'General inspection of building condition',
            'Frequency': 'Annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Check and clean gutters and drainage systems',
            'Frequency': 'Quarterly',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Inspect for pest activity',
            'Frequency': 'Semi-annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        }
    ]
    schedule_items.extend(standard_tasks)
    return schedule_items

def count_critical_issues(response_json: dict) -> Dict:
    """Extract critical issues counts from JSON"""
    non_compliant_count = sum(1 for item in response_json['detailedInspection'] 
                             if item['complianceStatus'] == 'Non-compliant')
    critical_issues_count = len(response_json['executiveSummary']['criticalIssues'])
    return {
        'critical_issues': non_compliant_count,
        'high_priority': critical_issues_count
    }
@app.callback(
    [Output('chat-output', 'children'),
     Output('chat-input', 'value')],
    [Input('send-button', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-output', 'children')],
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history):
    if not user_input:
        return no_update, no_update
    available_media = list(document_dict['user_data'].keys())
    print(f"Available media files: {available_media}")  
    response = chat_session.send_message(
        user_input + 
        ". Please provide a detailed answer with elaboration on the report and construction standards. " +
        "Format the response as a JSON object with a detailedInspection array. " +
        "Only reference these exact media files: " + 
        ", ".join(available_media)
    )

    try:
        response_data = json.loads(response.text)
        print(f"Response data: {json.dumps(response_data, indent=2)}")  
        if 'detailedInspection' in response_data:
            for item in response_data['detailedInspection']:
                if 'mediaReference' in item:
                    media_ref = item['mediaReference']
                    print(f"Checking media reference: {media_ref}") 
                    
                    if media_ref not in available_media:
                        print(f"Media reference not found: {media_ref}") 
                        for media in available_media:
                            if media.lower().startswith(media_ref.lower()):
                                item['mediaReference'] = media
                                print(f"Found matching media: {media}")  
                                break
                        else:
                            if available_media:
                                item['mediaReference'] = available_media[0]
                                print(f"Using first available media: {available_media[0]}")
            formatted_response = create_inspection_cards(response_data['detailedInspection'])
        else:
            formatted_response = format_text_response(response.text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}") 
        formatted_response = format_text_response(response.text)
    except Exception as e:
        print(f"Unexpected error: {e}")  
        formatted_response = format_text_response(str(e))
    new_messages = [
        html.Div([
            html.P("User:", style={'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '5px'}),
            html.P(user_input, style={'marginLeft': '20px', 'color': '#34495e'})
        ], style={'marginBottom': '15px'}),
        html.Div([
            html.P("Assistant:", style={'fontWeight': 'bold', 'color': '#2c3e50', 'marginBottom': '5px'}),
            html.Div(formatted_response, style={'marginLeft': '20px'})
        ], style={'marginBottom': '20px'})
    ]
    return (chat_history + new_messages if isinstance(chat_history, list) else new_messages), ''
def create_inspection_cards(inspection_items: List[Dict]) -> html.Div:
    """Create styled cards for inspection items with images"""
    cards = []
    for item in inspection_items:
        media_content = None
        if item.get('mediaReference'):
            try:
                media_ref = item['mediaReference']
                if media_ref.startswith('home_inspection.mp4') and item.get('timestamp'):
                    media_ref = f"home_inspection.mp4_{item['timestamp']}"
                relative_path = get_media_path(media_ref)
                full_path = os.path.join(base_dir, relative_path)
                if os.path.exists(full_path):
                    base64_img = get_image_base64(full_path)
                    if base64_img:
                        media_content = html.Div([
                            html.Img(
                                src=f"data:image/jpeg;base64,{base64_img}",
                                style={
                                    'width': '100%',
                                    'maxWidth': '500px',
                                    'height': 'auto',
                                    'maxHeight': '400px',
                                    'objectFit': 'contain',
                                    'display': 'block',
                                    'margin': '0 auto',
                                    'borderRadius': '8px',
                                }
                            )
                        ], style={
                            'width': '100%',
                            'textAlign': 'center',
                            'backgroundColor': '#f8f9fa',
                            'padding': '10px',
                            'borderRadius': '8px',
                            'marginBottom': '15px',
                        })
                        print(f"Created image component for: {media_ref}")  # Debug log
            except Exception as e:
                print(f"Error creating image component: {e}")  # Debug log
                media_content = None
        card = html.Div([
            html.Div([
                html.H4(item['area'], style={
                    'color': '#2c3e50',
                    'margin': '0',
                    'flex': '1'
                }),
                html.Span(
                    item['complianceStatus'],
                    style={
                        'backgroundColor': '#ffcccc' if item['complianceStatus'] == 'Non-compliant' else '#e8f5e9',
                        'color': '#990000' if item['complianceStatus'] == 'Non-compliant' else '#1b5e20',
                        'padding': '4px 8px',
                        'borderRadius': '4px',
                        'fontSize': '14px'
                    }
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '15px',
                'paddingBottom': '10px',
                'borderBottom': '2px solid #3498db'
            }),
            html.Div(
                media_content if media_content else html.Div(
                    f"No image available for: {item.get('mediaReference', 'None')}",
                    style={
                        'textAlign': 'center',
                        'padding': '20px',
                        'backgroundColor': '#f8f9fa',
                        'color': '#666',
                        'borderRadius': '8px',
                    }
                ),
                style={
                    'width': '100%',
                    'marginBottom': '15px',
                }
            ),
            html.Div([
                html.P([html.Strong("Condition: "), item.get('condition', 'N/A')]),
                html.P([html.Strong("Media Reference: "), 
                       f"{item.get('mediaReference', 'N/A')}" + 
                       (f"_{item['timestamp']}" if item.get('timestamp') and item.get('mediaReference', '').startswith('home_inspection.mp4') else "")]),
                html.P([html.Strong("Code Reference: "), 
                       f"{item.get('referenceDoc', 'N/A')} - {item.get('referenceSection', 'N/A')}"]),
                html.Div([
                    html.Strong("Issues Found:", style={'display': 'block', 'marginBottom': '8px'}),
                    html.Ul([
                        html.Li(issue, style={'marginBottom': '8px'}) 
                        for issue in item.get('issuesFound', [])
                    ], style={'paddingLeft': '20px'})
                ]) if item.get('issuesFound') else None,
                html.P([html.Strong("Recommendation: "), item.get('recommendation', 'N/A')])
            ], style={
                'color': '#2c3e50',
                'lineHeight': '1.6'
            })
        ], style={
            'backgroundColor': '#ffffff',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
            'border': '1px solid #e9ecef',
            'width': '100%',
        }) 
        cards.append(card)
    return html.Div(cards, style={'width': '100%'})
def format_structured_response(response_data: dict) -> List:
    """Format structured JSON responses with appropriate styling"""
    if not isinstance(response_data, dict):
        return format_text_response(str(response_data)) 
    formatted_sections = []
    for key, value in response_data.items():
        section = html.Div([
            html.H4(
                key.replace('_', ' ').title(),
                style={
                    'color': '#2c3e50',
                    'padding': '10px 15px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px 8px 0 0',
                    'margin': '0',
                    'borderBottom': '2px solid #3498db'
                }
            ),
            html.Div(
                format_content(value),
                style={
                    'padding': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '0 0 8px 8px'
                }
            )
        ], style={
            'marginBottom': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'border': '1px solid #e9ecef'
        })
        formatted_sections.append(section)
    return formatted_sections
def format_content(value) -> html.Div:
    """Format content based on its type"""
    if isinstance(value, dict):
        return format_dict_content(value)
    elif isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return format_dict_list(value)
        return format_list_content(value)
    else:
        return html.P(
            str(value),
            style={
                'margin': '0',
                'color': '#2c3e50'
            }
        )
def format_dict_content(data: dict) -> html.Div:
    """Format dictionary content with improved styling"""
    items = []
    for key, value in data.items():
        item = html.Div([
            html.Div(
                key.replace('_', ' ').title() + ':',
                style={
                    'fontWeight': 'bold',
                    'color': '#34495e',
                    'backgroundColor': '#f8f9fa',
                    'padding': '8px 12px',
                    'borderRadius': '4px',
                    'marginBottom': '5px'
                }
            ),
            html.Div(
                format_content(value),
                style={
                    'marginLeft': '12px',
                    'marginBottom': '15px',
                    'color': '#2c3e50'
                }
            )
        ])
        items.append(item)
    return html.Div(items)
def format_dict_list(data_list: List[dict]) -> html.Div:
    """Format a list of dictionaries"""
    return html.Div([
        html.Div(
            format_dict_content(item),
            style={
                'backgroundColor': 'white',
                'padding': '15px',
                'borderRadius': '8px',
                'marginBottom': '10px',
                'border': '1px solid #e9ecef'
            }
        ) for item in data_list
    ])
def format_list_content(data: list) -> html.Div:
    """Format list content with enhanced styling"""
    return html.Ul([
        html.Li(
            format_content(item) if isinstance(item, (dict, list)) else str(item),
            style={
                'marginBottom': '8px',
                'color': '#2c3e50',
                'lineHeight': '1.5'
            }
        ) for item in data
    ], style={
        'listStyleType': 'disc',
        'paddingLeft': '20px',
        'margin': '0'
    })
def format_text_response(text: str) -> List:
    """Format plain text responses with enhanced styling"""
    try:
       
        data = json.loads(text)
        if 'detailedInspection' in data:
            return create_inspection_cards(data['detailedInspection'])
    except json.JSONDecodeError:
        pass
    paragraphs = text.split('\n\n')
    
    formatted_paragraphs = []
    for para in paragraphs:
        if para.strip().startswith('#'):
            level = len(para.split()[0])
            text = ' '.join(para.split()[1:])
            formatted_paragraphs.append(html.H4(text, style={
                'color': '#2c3e50',
                'marginTop': '20px',
                'marginBottom': '10px',
                'fontSize': f'{24 - (level * 2)}px'
            }))
        elif para.strip().startswith('- ') or para.strip().startswith('* '):
            items = [item.strip('- *') for item in para.split('\n')]
            formatted_paragraphs.append(html.Ul([
                html.Li(item, style={'marginBottom': '5px'}) 
                for item in items
            ], style={'marginLeft': '20px'}))
        else:
            formatted_paragraphs.append(html.P(para, style={
                'lineHeight': '1.6',
                'color': '#2c3e50',
                'marginBottom': '15px'
            }))
    return [html.Div(formatted_paragraphs, style={
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })]
inspection_data = pd.DataFrame(parse_inspection_table(response_json))
maintenance_schedule = pd.DataFrame(parse_maintenance_schedule(response_json))
issue_counts = count_critical_issues(response_json)
compliance_counts = inspection_data['Priority'].value_counts()
app.layout = html.Div([
    html.H1('Building Inspection Report Dashboard', 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    html.Div([
        html.Div([
            html.H3('Critical Issues'),
            html.P(f"{issue_counts['critical_issues']} Non-Compliant Areas"),
            html.P(f"{issue_counts['high_priority']} High Priority Items")
        ], className='summary-card'),
        html.Div([
            html.H3('Maintenance Tasks'),
            html.P(f"{len(maintenance_schedule)} Total Tasks"),
            html.P(f"{len(maintenance_schedule[maintenance_schedule['Priority'] == 'High']) if 'Priority' in maintenance_schedule.columns else 0} High Priority Tasks")
        ], className='summary-card'),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
    html.Div([
        html.Div([
            html.H2('Inspection Results', style={'display': 'inline-block', 'marginRight': '10px'}),
            html.Button(
                '▼',
                id='toggle-inspection-button',
                style={
                    'backgroundColor': 'transparent',
                    'border': 'none',
                    'fontSize': '20px',
                    'cursor': 'pointer',
                    'padding': '5px',
                    'verticalAlign': 'middle'
                }
            )
        ], style={'marginBottom': '10px'}),
        
        html.Div(
            dash_table.DataTable(
                id='inspection-table',
                data=inspection_data.to_dict('records'),
                columns=[
                    {'name': 'Area', 'id': 'Area'},
                    {
                        'name': 'Media',
                        'id': 'Media',
                        'presentation': 'markdown',
                        'type': 'text'
                    },
                    {
                        'name': 'Details',
                        'id': 'Details',
                        'presentation': 'markdown',
                        'type': 'text'
                    },
                    {'name': 'Priority', 'id': 'Priority'}
                ],
                markdown_options={
                    'html': True,
                    'link_target': '_blank'
                },
                style_cell={
                    'textAlign': 'left',
                    'padding': '15px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '100px',
                    'maxWidth': '400px',
                },
                style_header={
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'height': '40px',  
                    'lineHeight': '40px', 
                    'padding': '0 15px',  
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Media'},
                        'width': '500px',
                        'height': '400px',
                        'padding': '0px',
                    },
                    {
                        'if': {'column_id': 'Details'},
                        'width': '400px',
                        'whiteSpace': 'pre-line',
                    }
                ],
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Condition', 'filter_query': '{Condition} eq "Non-Compliant"'},
                        'backgroundColor': '#ffcccc',
                        'color': '#990000'
                    },
                    {
                        'if': {'column_id': 'Priority', 'filter_query': '{Priority} eq "High"'},
                        'backgroundColor': '#ffcccc',
                        'color': '#990000'
                    }
                ],
                css=[{
                    'selector': '.dash-cell-value',
                    'rule': 'display: flex; align-items: center; justify-content: center;'
                }]
            ),
            id='inspection-container',
            style={'display': 'block'}
        )
    ], style={'marginBottom': 30}),
    html.Div([
        html.H2('Maintenance Schedule'),
        dash_table.DataTable(
            data=maintenance_schedule.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in maintenance_schedule.columns],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Priority', 'filter_query': '{Priority} eq "High"'},
                    'backgroundColor': '#ffcccc',
                    'color': '#990000'
                }
            ],
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Task'},
                    'width': '40%', 
                    'minWidth': '200px',  
                    'maxWidth': '400px' 
                },
                {
                    'if': {'column_id': 'Frequency'},
                    'width': '20%'
                },
                {
                    'if': {'column_id': 'Priority'},
                    'width': '20%'
                },
                {
                    'if': {'column_id': 'Status'},
                    'width': '20%'
                }
            ]
        )
    ], style={'marginBottom': 30}),
    html.Div([
        html.H2('Ask Questions About the Inspection'),
        html.Div([
            dcc.Input(
                id='chat-input',
                type='text',
                placeholder='Ask a question about the inspection...',
                style={
                    'width': '80%',
                    'padding': '10px',
                    'marginRight': '10px',
                    'borderRadius': '5px',
                    'border': '1px solid #ddd'
                }
            ),
            html.Button('Send', 
                id='send-button',
                style={
                    'padding': '10px 20px',
                    'backgroundColor': '#2c3e50',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            )
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        dls.Hash(
            html.Div(id='chat-output',
                style={
                    'maxHeight': '600px', 
                    'height': '600px',    
                    'overflowY': 'auto',
                    'padding': '20px',
                    'backgroundColor': 'white',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                }
            ),
            color="#2c3e50",
            debounce=300,
            id='chat-loading'
        )
    ], style={'margin': '20px 0', 'width': '100%'})
], style={'padding': '20px', 'fontFamily': 'Arial'})
from typing import List, Dict, Union
import google.generativeai as genai
def get_gemini_response(question: str, context: str) -> str:
    """Get response from Gemini model using chat functionality"""
    response = chat_session.send_message(question + ". Please provide a detailed answer with elaboration on the report and user provided material.")
    return response.text
@app.callback(
    [Output('inspection-container', 'style'),
     Output('toggle-inspection-button', 'children')],
    [Input('toggle-inspection-button', 'n_clicks')],
    [State('inspection-container', 'style')]
)
def toggle_inspection_table(n_clicks, current_style):
    if n_clicks is None:
        return current_style, '▼'
    if current_style.get('display') == 'none':
        return {'display': 'block'}, '▼' 
    else:
        return {'display': 'none'}, '▶' 
@app.callback(
    Output('inspection-table', 'data'),
    Input('inspection-table', 'data')
)
def update_table_data(data):
    if not data:
        return []     
    for row in data:
        media_ref = row['Media']
        if media_ref.startswith('home_inspection.mp4') and '_' in media_ref:
            media_key = media_ref
        else:
            media_key = media_ref
        relative_path = get_media_path(media_key)
        full_path = os.path.join(base_dir, relative_path)
        if os.path.exists(full_path):
            base64_img = get_image_base64(full_path)
            if base64_img:
                row['Media'] = f'<img src="data:image/jpeg;base64,{base64_img}" style="width: 100%; height: 100%; object-fit: contain; display: block;">'
        else:
            print(f"File not found: {full_path}")
            row['Media'] = f"Image not found: {media_key}"
    return data
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Building Inspection Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .summary-card {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 250px;
                text-align: center;
            }
            .summary-card h3 {
                color: #2c3e50;
                margin-bottom: 15px;
            }
            body {
                background-color: #f5f6fa;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
if __name__ == '__main__':
    print("Dash app running on: http://127.0.0.1:8051")
    app.run(
        port=8051,         # New port (instead of 8050)
        debug=True,
        host="127.0.0.1"   # Explicitly say localhost
    )
