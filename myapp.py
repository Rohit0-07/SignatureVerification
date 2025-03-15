import streamlit as st
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import pandas as pd 
from PIL import Image
import io
import json
import hashlib
import datetime
import time
import sys
import matplotlib.pyplot as plt
import concurrent.futures
from google.cloud import vision
from google.cloud import storage
from google.cloud import firestore
import google.generativeai as genai

# Global ORB detector to reuse across comparisons for performance
ORB_DETECTOR = cv2.ORB_create(nfeatures=500)

st.set_page_config(
    page_title="Signature Verification System",
    page_icon="✅",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'last_operation_time' not in st.session_state:
    st.session_state.last_operation_time = None
if 'operation_durations' not in st.session_state:
    st.session_state.operation_durations = []

###############################################################################
# Performance Monitoring
###############################################################################
import threading

def timed_operation(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        # Only update session state if in the main thread
        if threading.current_thread() == threading.main_thread():
            st.session_state.last_operation_time = duration
            st.session_state.operation_durations.append({
                "operation": func.__name__,
                "duration": duration,
                "timestamp": datetime.datetime.now()
            })
        else:
            # Optionally log or handle non-main thread timings differently
            print(f"Operation {func.__name__} took {duration:.4f} seconds in a non-main thread")
            
        return result
    return wrapper

###############################################################################
# Helper Functions for Image Processing and Signature Comparison
###############################################################################
@st.cache_data(ttl=300)
def preprocess_image(image):
    """Preprocess image: grayscale, binary threshold, and morphological opening."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
    else:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary

@st.cache_data(ttl=300)
def center_signature(binary_image, canvas_size=(300,150)):
    """
    Crop the signature to its bounding box and resize to a consistent canvas.
    """
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(binary_image)
    if coords is None:
        return cv2.resize(binary_image, canvas_size, interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary_image[y:y+h, x:x+w]
    resized = cv2.resize(cropped, canvas_size, interpolation=cv2.INTER_AREA)
    return resized

@timed_operation
def compare_signatures(signature1, signature2):
    """
    Compare two signatures using ORB keypoints and SSIM on centered images.
    Returns a score between 0 and 1 and a visualization of the comparison.
    """
    sig1 = center_signature(signature1)
    sig2 = center_signature(signature2)
    
    # SSIM comparison
    ssim_score = ssim(sig1, sig2, data_range=255)
    
    # ORB keypoints matching using a global detector for speed
    kp1, des1 = ORB_DETECTOR.detectAndCompute(sig1, None)
    kp2, des2 = ORB_DETECTOR.detectAndCompute(sig2, None)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        orb_score = 0.0
        comparison_image = np.hstack((sig1, sig2))
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) == 0:
            orb_score = 0.0
            comparison_image = np.hstack((sig1, sig2))
        else:
            matches = sorted(matches, key=lambda x: x.distance)
            avg_dist = np.mean([m.distance for m in matches])
            maxDist = 200.0  # normalization factor (adjustable)
            orb_score = max(0.0, 1.0 - (avg_dist / maxDist))
            
            # Create comparison visualization
            comparison_image = cv2.drawMatches(
                sig1, kp1, sig2, kp2, matches[:10], None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
    
    # Combine scores with equal weight
    final_score = 0.5 * ssim_score + 0.5 * orb_score
    return final_score, comparison_image

###############################################################################
# Google Cloud Setup and Authentication
###############################################################################
@st.cache_resource
def setup_google_cloud():
    """Set up Google Cloud services with caching to improve performance."""
    credentials = dict(st.secrets["gcp_service_account"])
    credentials_path = "credentials.json"
    with open(credentials_path, "w") as f:
        json.dump(credentials, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    db = firestore.Client()
    api_key = credentials.get("api_key", "")
    if api_key:
        genai.configure(api_key=api_key)
    return vision_client, storage_client, db

try:
    vision_client, storage_client, db = setup_google_cloud()
    st.sidebar.success("Successfully connected to Google Cloud!")
except Exception as e:
    st.sidebar.error(f"Failed to connect to Google Cloud: {str(e)}")
    st.stop()

@st.cache_resource
def ensure_bucket_exists(bucket_name="signature_verification_bucket"):
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        bucket = storage_client.create_bucket(bucket_name)
        st.sidebar.info(f"Created new bucket: {bucket_name}")
    return bucket

signature_bucket = ensure_bucket_exists()

def authenticate(username, password):
    """Authenticate user against admin credentials or Firestore database."""
    try:
        if username == st.secrets["admin"]["username"] and password == st.secrets["admin"]["password"]:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.is_admin = True
            st.session_state.user_role = "Administrator"  # Set role for admin
            # Log successful admin login
            log_activity(username, "admin_login", "Admin user logged in")
            return True

        users_ref = db.collection('users')
        user_doc = users_ref.document(username).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            stored_password_hash = user_data.get('password_hash')
            input_password_hash = hashlib.sha256(password.encode()).hexdigest()
            if stored_password_hash == input_password_hash:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.is_admin = False
                # Store user role from metadata (defaulting to Standard User if not set)
                st.session_state.user_role = user_data.get('metadata', {}).get('role', 'Standard User')
                # Log successful user login
                log_activity(username, "user_login", "User logged in")
                return True
        # Log failed login attempt
        log_activity(username, "failed_login", "Failed login attempt")
        return False
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False


def logout():
    """Log out the current user and clear session state."""
    if st.session_state.authenticated:
        log_activity(st.session_state.username, "logout", "User logged out")
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.is_admin = False

###############################################################################
# Signature Extraction (with Gemini and fallback to contour detection)
###############################################################################
@timed_operation
def extract_signature(image):
    """Extract signature from document image using Gemini AI with fallback to contour detection."""
    if image is None or image.size == 0:
        return None, "Invalid image input"
    
    # Ensure image is in proper format
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Prepare image for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Try Gemini AI for signature extraction
    try:
        success, img_encoded = cv2.imencode('.jpg', image)
        if not success:
            return None, "Failed to encode image"
        
        # Prepare image for Gemini
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # Use Gemini for signature detection
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = """
        This is a document image. Please identify if there is a signature in this image.
        If there is, provide the approximate bounding box coordinates of the signature 
        in the format: [x_min, y_min, x_width, y_height] where values are relative to 
        the image dimensions (0 to 1). If no signature is found, respond with "No signature found".
        """
        
        try:
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ])
        except Exception as first_error:
            try:
                response = model.generate_content(
                    contents=[{"text": prompt}, {"image": {"data": img_bytes, "mime_type": "image/jpeg"}}]
                )
            except Exception as second_error:
                st.warning("Gemini API failed. Falling back to contour detection.")
                return fallback_signature_extraction(gray)
        
        if "No signature found" in response.text:
            st.warning("No signature found by Gemini. Trying contour detection method.")
            return fallback_signature_extraction(gray)
        
        import re
        coord_match = re.search(r'\[([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+),\s*([0-9\.]+)\]', response.text)
        if coord_match:
            x_min = float(coord_match.group(1))
            y_min = float(coord_match.group(2))
            x_width = float(coord_match.group(3))
            y_height = float(coord_match.group(4))
            height, width = image.shape[:2]
            x_min_abs = int(x_min * width)
            y_min_abs = int(y_min * height)
            x_width_abs = int(x_width * width)
            y_height_abs = int(y_height * height)
            
            # Ensure valid coordinates
            x_min_abs = max(0, x_min_abs)
            y_min_abs = max(0, y_min_abs)
            x_width_abs = min(width - x_min_abs, x_width_abs)
            y_height_abs = min(height - y_min_abs, y_height_abs)
            
            if x_width_abs <= 0 or y_height_abs <= 0:
                st.warning("Invalid signature region detected. Trying contour detection method.")
                return fallback_signature_extraction(gray)
            
            signature = image[y_min_abs:y_min_abs+y_height_abs, x_min_abs:x_min_abs+x_width_abs]
            processed_signature = preprocess_image(signature)
            return processed_signature, "Signature extracted successfully with Gemini AI"
        else:
            st.warning("Gemini couldn't provide coordinates. Falling back to contour detection.")
            return fallback_signature_extraction(gray)
    except Exception as e:
        st.warning(f"Error using Gemini: {str(e)}. Falling back to contour detection.")
        return fallback_signature_extraction(gray)

def fallback_signature_extraction(gray_image):
    """Fallback method to extract signature using contour detection."""
    try:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area, aspect ratio, and other heuristics
        potential_signatures = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Signature typically has reasonable size and aspect ratio
            if area > 200 and w > 50 and h > 20 and aspect_ratio < 7:
                potential_signatures.append((x, y, w, h, area))
        
        if not potential_signatures:
            return None, "No potential signatures found"
        
        # Sort by area in descending order
        potential_signatures.sort(key=lambda x: x[4], reverse=True)
        
        # Take the largest contour by area which is likely the signature
        x, y, w, h, _ = potential_signatures[0]
        signature = gray_image[y:y+h, x:x+w]
        processed_signature = preprocess_image(signature)
        return processed_signature, "Signature extracted using contour detection"
    except Exception as e:
        return None, f"Error in contour detection: {str(e)}"

###############################################################################
# Activity Logging
###############################################################################
def log_activity(username, action_type, description):
    """Log user activity to Firestore for audit trail."""
    try:
        log_ref = db.collection('activity_logs').document()
        log_ref.set({
            "username": username,
            "action_type": action_type,
            "description": description,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "ip_address": "N/A"  # In production, you'd capture the actual IP
        })
        return True
    except Exception as e:
        print(f"Error logging activity: {str(e)}")
        return False

###############################################################################
# Signature Verification and Comparison Across Users
###############################################################################
@timed_operation
def verify_signature(upload_signature, username, threshold=0.5):
    """Verify a signature against a specific user's stored signatures."""
    if not username:
        return {"is_genuine": False, "confidence": 0, "threshold": threshold}, "No username provided", None
    
    user_ref = db.collection('users').document(username)
    if not user_ref.get().exists:
        return {"is_genuine": False, "confidence": 0, "threshold": threshold}, f"User '{username}' does not exist", None
    
    blobs = list(storage_client.list_blobs(signature_bucket.name, prefix=f"signatures/{username}/"))
    
    if not blobs:
        return {"is_genuine": False, "confidence": 0, "threshold": threshold}, f"No signatures stored for user '{username}'", None
    
    best_match_score = 0
    best_match_path = None
    best_comparison_image = None
    
    for blob in blobs:
        try:
            signature_data = blob.download_as_bytes()
            signature_array = np.frombuffer(signature_data, np.uint8)
            stored_signature = cv2.imdecode(signature_array, cv2.IMREAD_GRAYSCALE)
            
            if stored_signature is None or upload_signature is None:
                continue
                
            similarity_score, comparison_image = compare_signatures(upload_signature, stored_signature)
            
            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match_path = blob.name
                best_comparison_image = comparison_image
        except Exception as e:
            print(f"Error processing {blob.name}: {str(e)}")
            continue
    
    is_genuine = best_match_score >= threshold
    
    # Log verification attempt
    log_activity(username, "signature_verification", 
                 f"Verification {'successful' if is_genuine else 'failed'} with confidence {best_match_score:.2f}")
    
    return {
        "is_genuine": is_genuine,
        "confidence": best_match_score,
        "best_match": best_match_path,
        "threshold": threshold
    }, f"Verification complete for user '{username}'", best_comparison_image

@timed_operation
def verify_signature_across_all_users(upload_signature, threshold=0.5):
    """Verify a signature against all users in the system."""
    all_users = list_users()
    best_score = 0.0
    best_user = None
    best_blob = None
    best_comparison_image = None
    
    # Use ThreadPoolExecutor for parallel processing
    def check_user(user_info):
        username = user_info["username"]
        user_best_score = 0.0
        user_best_blob = None
        user_best_image = None
        
        blobs = list(storage_client.list_blobs(signature_bucket.name, prefix=f"signatures/{username}/"))
        for blob in blobs:
            try:
                signature_data = blob.download_as_bytes()
                signature_array = np.frombuffer(signature_data, np.uint8)
                stored_signature = cv2.imdecode(signature_array, cv2.IMREAD_GRAYSCALE)
                
                if stored_signature is None or upload_signature is None:
                    continue
                    
                score, comparison_image = compare_signatures(upload_signature, stored_signature)
                
                if score > user_best_score:
                    user_best_score = score
                    user_best_blob = blob.name
                    user_best_image = comparison_image
            except Exception as e:
                print(f"Error processing {blob.name} for user {username}: {str(e)}")
                continue
        
        return username, user_best_score, user_best_blob, user_best_image
    
    # Process users in parallel (adjust max_workers based on your system)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(check_user, all_users))
    
    # Find the best match across all users
    for username, score, blob, image in results:
        if score > best_score:
            best_score = score
            best_user = username
            best_blob = blob
            best_comparison_image = image
    
    found_match = best_score >= threshold
    
    # Log cross-user verification
    if found_match:
        log_activity("system", "cross_user_verification", 
                     f"Found match with user {best_user} with confidence {best_score:.2f}")
    else:
        log_activity("system", "cross_user_verification", 
                     f"No matches found. Best match was user {best_user} with confidence {best_score:.2f}")
    
    return found_match, best_user, best_score, best_blob, best_comparison_image

###############################################################################
# Firestore User Management
###############################################################################
@st.cache_data(ttl=60)
def list_users():
    """List all users in the system."""
    users = []
    users_ref = db.collection('users')
    for user in users_ref.stream():
        user_data = user.to_dict()
        users.append({
            "username": user.id,
            "created_at": user_data.get("created_at"),
            "metadata": user_data.get("metadata", {})
        })
    return users

def get_user_metadata(username):
    """Get metadata for a specific user."""
    user_ref = db.collection('users').document(username)
    user_doc = user_ref.get()
    if user_doc.exists:
        return user_doc.to_dict()
    return None

def add_user(username, password, metadata):
    """Add a new user to the system."""
    if not username or not password:
        return False, "Username and password are required"
    
    try:
        user_ref = db.collection('users').document(username)
        if user_ref.get().exists:
            return False, "User already exists"
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user_data = {
            "username": username,
            "password_hash": password_hash,
            "created_at": firestore.SERVER_TIMESTAMP,
            "metadata": metadata
        }
        user_ref.set(user_data)
        
        # Log user creation
        log_activity("admin", "user_created", f"Created user: {username}")
        
        return True, "User added successfully"
    except Exception as e:
        return False, f"Error adding user: {str(e)}"

def upload_signature(username, signature_image):
    """Upload a signature for a user."""
    if not username:
        return False, "Username is required"
    
    try:
        is_success, buffer = cv2.imencode(".png", signature_image)
        if not is_success:
            return False, "Failed to encode image"
        
        byte_stream = io.BytesIO(buffer)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"signatures/{username}/{timestamp}.png"
        blob = signature_bucket.blob(blob_name)
        blob.upload_from_file(byte_stream, content_type="image/png")
        
        # Log signature upload
        log_activity(username, "signature_uploaded", f"Uploaded signature: {blob_name}")
        
        return True, f"Signature uploaded successfully as {blob_name}"
    except Exception as e:
        return False, f"Error uploading signature: {str(e)}"

def delete_user(username):
    """Delete a user and all their associated signatures."""
    if not username:
        return False, "Username is required"
    
    try:
        # Check if user exists
        user_ref = db.collection('users').document(username)
        if not user_ref.get().exists:
            return False, "User does not exist"
        
        # Delete user document from Firestore
        user_ref.delete()
        
        # Delete user signatures from Storage bucket
        blobs = list(storage_client.list_blobs(signature_bucket.name, prefix=f"signatures/{username}/"))
        for blob in blobs:
            blob.delete()
        
        # Log user deletion
        log_activity("admin", "user_deleted", f"Deleted user: {username}")
        
        return True, f"User {username} and all associated signatures deleted successfully"
    except Exception as e:
        return False, f"Error deleting user: {str(e)}"

###############################################################################
# Streamlit Interface
###############################################################################
def main():
    st.title("Signature Verification System")
    st.sidebar.title("Navigation")
    
    if st.session_state.authenticated:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        
        if st.session_state.is_admin:
            page = st.sidebar.radio("Select Page", ["Admin Dashboard", "Verify Signature", "Manage Users", "System Performance"])
        else:
            page = "Verify Signature"
            
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
    else:
        page = "Login"
        
    if page == "Login":
        display_login_page()
    elif page == "Admin Dashboard":
        display_admin_dashboard()
    elif page == "Verify Signature":
        display_verification_page()
    elif page == "Manage Users":
        display_manage_users_page()
    elif page == "System Performance":
        display_performance_page()

def display_login_page():
    st.header("Login")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            with st.spinner("Authenticating..."):
                if authenticate(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

def display_admin_dashboard():
    st.header("Admin Dashboard")
    
    # System metrics
    st.subheader("System Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        users = list_users()
        st.metric("Total Users", len(users))
    
    with col2:
        total_signatures = 0
        blobs = list(storage_client.list_blobs(signature_bucket.name, prefix="signatures/"))
        for _ in blobs:
            total_signatures += 1
        st.metric("Total Signatures", total_signatures)
    
    with col3:
        # Get recent verification counts
        try:
            logs_ref = db.collection('activity_logs')
            query = logs_ref.filter('action_type', '==', 'signature_verification').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100)
            verification_logs = list(query.stream())
            st.metric("Recent Verifications", len(verification_logs))
        except Exception:
            st.metric("Recent Verifications", "N/A")
    
    # Recent activity chart
    st.subheader("Recent Activity")
    try:
        logs_ref = db.collection('activity_logs')
        query = logs_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50)
        logs = list(query.stream())
        
        if logs:
            log_data = []
            for log in logs:
                log_dict = log.to_dict()
                log_data.append({
                    "Username": log_dict.get("username", "Unknown"),
                    "Action": log_dict.get("action_type", "Unknown"),
                    "Description": log_dict.get("description", ""),
                    "Timestamp": log_dict.get("timestamp", datetime.datetime.now())
                })
            
            df_logs = pd.DataFrame(log_data)
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No recent activity logs found")
    except Exception as e:
        st.error(f"Error retrieving activity logs: {str(e)}")
    
    # Recent users
    st.subheader("Users")
    users_sorted = sorted(list_users(), key=lambda x: x.get("created_at", datetime.datetime.min), reverse=True)
    
    if users_sorted:
        with st.expander("View All Users", expanded=False):
            for user in users_sorted:
                st.markdown(f"**User:** {user['username']}")
                st.write(f"Created: {user.get('created_at', 'Unknown')}")
                st.write("Metadata:")
                st.json(user.get("metadata", {}))
                st.markdown("---")
    else:
        st.info("No users in the system yet")

def display_verification_page():
    st.header("Signature Verification")
    
    # Slider for user-adjustable threshold
    threshold = st.slider(
        "Set Verification Threshold", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Adjust the matching threshold for signature verification (higher = stricter)"
    )
    
    upload_mode = st.radio("Upload Mode", ["Document with Signature", "Signature Only"])
    verify_mode = st.radio("Verification Mode", ["Single User", "All Users"], key="verification_mode")
    
    uploaded_files = st.file_uploader(
        "Upload Document/Signature", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    # Process and store all uploaded files
    all_processed_signatures = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Display the file title
            st.markdown(f"### File: {uploaded_file.name}")
            image = Image.open(uploaded_file)
            image_cv = np.array(image)
            if len(image_cv.shape) == 3:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            process_key = f"process_{uploaded_file.name}"
            if st.button("Process Image", key=process_key):
                with st.spinner("Processing..."):
                    if upload_mode == "Document with Signature":
                        signature, message = extract_signature(image_cv)
                        if signature is None:
                            st.error(message)
                        else:
                            st.success(message)
                            st.image(signature, caption="Extracted Signature", use_container_width=True)
                            st.session_state[f"processed_{uploaded_file.name}"] = signature
                            all_processed_signatures[uploaded_file.name] = signature
                    else:
                        signature = preprocess_image(image_cv)
                        st.image(signature, caption="Processed Signature", use_container_width=True)
                        st.session_state[f"processed_{uploaded_file.name}"] = signature
                        all_processed_signatures[uploaded_file.name] = signature
            
            # Check if the signature has been processed
            if f"processed_{uploaded_file.name}" in st.session_state:
                signature = st.session_state[f"processed_{uploaded_file.name}"]
                all_processed_signatures[uploaded_file.name] = signature
                
                # Display verification options for Single User mode
                if verify_mode == "Single User":
                    username = st.text_input(
                        f"Enter username to verify against", key=f"username_{uploaded_file.name}"
                    )
                    
                    verify_key = f"verify_{uploaded_file.name}"
                    if st.button("Verify Signature", key=verify_key):
                        if not username:
                            st.error("Please enter a username")
                        else:
                            with st.spinner("Verifying signature..."):
                                # First check if the user exists
                                user_ref = db.collection('users').document(username)
                                if not user_ref.get().exists:
                                    st.error(f"User '{username}' does not exist. Cannot verify.")
                                else:
                                    result, message, comparison_image = verify_signature(signature, username, threshold)
                                    
                                    if result["is_genuine"]:
                                        st.success(
                                            f"✅ Signature VERIFIED with {result['confidence']:.2f} confidence (threshold: {threshold})"
                                        )
                                    else:
                                        st.error(
                                            f"❌ Signature NOT VERIFIED. Confidence: {result['confidence']:.2f} (threshold: {threshold})"
                                        )
                                    
                                    st.write(message)
                                    
                                    if comparison_image is not None:
                                        st.image(comparison_image, caption="Signature Comparison", use_container_width=True)
                
                # Save signature option
                with st.expander("Save this signature"):
                    if st.session_state.is_admin:
                        save_username = st.text_input(
                            f"Save signature for username", key=f"save_username_{uploaded_file.name}"
                        )
                        if st.button("Save Signature", key=f"save_{uploaded_file.name}"):
                            if not save_username:
                                st.error("Please enter a username")
                            else:
                                user_ref = db.collection('users').document(save_username)
                                if not user_ref.get().exists:
                                    st.error(f"User {save_username} does not exist")
                                else:
                                    success, message = upload_signature(save_username, signature)
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                    else:
                        # Regular users can only save to their own account
                        if st.button("Save to My Account", key=f"save_own_{uploaded_file.name}"):
                            success, message = upload_signature(st.session_state.username, signature)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
    
        # Display a single "Verify Against All Users" button outside the loop
        if verify_mode == "All Users" and all_processed_signatures:
            st.markdown("---")
            st.subheader("Verify Against All Users")
            
            # Let user select which signature to verify
            signature_names = list(all_processed_signatures.keys())
            selected_signature = st.selectbox("Select signature to verify", signature_names)
            
            verify_all_key = "verify_all_users"
            if st.button("Verify Against All Users", key=verify_all_key):
                signature = all_processed_signatures[selected_signature]
                with st.spinner("Verifying signature against all users..."):
                    found_match, best_user, best_score, best_blob, best_comparison_image = verify_signature_across_all_users(signature, threshold)
                    
                    if found_match:
                        st.success(
                            f"✅ Signature MATCHED with user '{best_user}' with {best_score:.2f} confidence (threshold: {threshold})"
                        )
                        if best_comparison_image is not None:
                            st.image(best_comparison_image, caption=f"Best Match: {best_user}", use_container_width=True)
                    else:
                        st.warning(
                            f"❌ No matching signatures found above threshold. Best match: {best_user} with {best_score:.2f} confidence (threshold: {threshold})"
                        )
                        
def display_manage_users_page():
    # Only allow access if the current user is an Administrator or Manager
    if not st.session_state.is_admin and st.session_state.get("user_role", "Standard User") not in ["Manager", "Administrator"]:
        st.error("Admin or Manager access required")
        return

    st.header("User Management")
    
    # Define tabs. The "Add Signature" tab is added for admins/managers.
    tabs = ["Add User", "View Users", "Delete User", "Add Signature"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # ----- Tab: Add User -----
    with tab1:
        st.subheader("Add New User")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        
        # Additional metadata
        with st.expander("User Metadata"):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            department = st.text_input("Department")
            role = st.selectbox("Role", ["Standard User", "Manager", "Administrator"])
            
            metadata = {
                "full_name": full_name,
                "email": email,
                "department": department,
                "role": role,
                "created_by": st.session_state.username
            }
        
        if st.button("Add User"):
            if not new_username or not new_password:
                st.error("Username and password are required")
            else:
                with st.spinner("Adding user..."):
                    success, message = add_user(new_username, new_password, metadata)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # ----- Tab: View Users -----
    with tab2:
        st.subheader("Existing Users")
        users = list_users()
        
        if not users:
            st.info("No users found")
        else:
            # Prepare a DataFrame for display
            user_data = []
            for user in users:
                user_info = {
                    "Username": user["username"],
                    "Created At": user.get("created_at", "Unknown")
                }
                # Add metadata fields (excluding sensitive info)
                metadata = user.get("metadata", {})
                for key, value in metadata.items():
                    if key not in ["password_hash"]:
                        user_info[key] = value
                user_data.append(user_info)
            
            df_users = pd.DataFrame(user_data)
            st.dataframe(df_users, use_container_width=True)
            
            # Option to view details for a selected user
            selected_username = st.selectbox("Select user to view details", [user["username"] for user in users])
            if selected_username:
                st.subheader(f"User: {selected_username}")
                # Fetch user signatures count
                signature_count = 0
                blobs = list(storage_client.list_blobs(signature_bucket.name, prefix=f"signatures/{selected_username}/"))
                for _ in blobs:
                    signature_count += 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Registered Signatures", signature_count)
                with col2:
                    try:
                        logs_ref = db.collection('activity_logs')
                        query = logs_ref.filter('username', '==', selected_username).filter('action_type', '==', 'signature_verification')
                        verification_count = len(list(query.stream()))
                        st.metric("Verification Attempts", verification_count)
                    except Exception:
                        st.metric("Verification Attempts", "N/A")
                
                if signature_count > 0:
                    st.subheader("Registered Signatures")
                    signature_grid = st.columns(min(4, max(1, signature_count)))
                    for i, blob in enumerate(blobs):
                        col_idx = i % len(signature_grid)
                        with signature_grid[col_idx]:
                            try:
                                signature_data = blob.download_as_bytes()
                                signature_array = np.frombuffer(signature_data, np.uint8)
                                signature = cv2.imdecode(signature_array, cv2.IMREAD_GRAYSCALE)
                                st.image(signature, caption=f"Signature {i+1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error loading signature: {str(e)}")
    
    # ----- Tab: Delete User -----
    with tab3:
        st.subheader("Delete User")
        st.warning("⚠️ This action cannot be undone. All user data and signatures will be permanently deleted.")
        
        delete_username = st.text_input("Username to delete")
        confirm_delete = st.checkbox("I confirm that I want to delete this user and all their data")
        
        if st.button("Delete User", disabled=not confirm_delete):
            if not delete_username:
                st.error("Please enter a username")
            elif not confirm_delete:
                st.error("Please confirm the deletion")
            else:
                with st.spinner("Deleting user..."):
                    try:
                        success, message = delete_user(delete_username)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Error deleting user: {str(e)}")
                        log_activity("system", "error", f"Error in delete_user: {str(e)}")
    
    # ----- Tab: Add Signature -----
    with tab4:
        st.subheader("Add User Signatures")
        users = list_users()
        if not users:
            st.info("No users available")
        else:
            selected_user = st.selectbox("Select a user", [user["username"] for user in users], key="add_sig_select")
            uploaded_signatures = st.file_uploader(
                "Upload signature images", 
                type=["jpg", "jpeg", "png"], 
                key="add_signature_uploader", 
                accept_multiple_files=True
            )
            
            processed_signatures = []
            if uploaded_signatures:
                st.write("Preview of processed signatures:")
                for idx, uploaded_signature in enumerate(uploaded_signatures):
                    image = Image.open(uploaded_signature)
                    image_cv = np.array(image)
                    if len(image_cv.shape) == 3:
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                    processed_signature = preprocess_image(image_cv)
                    processed_signatures.append(processed_signature)
                    st.image(processed_signature, caption=f"Processed Signature {idx+1}", use_container_width=True)
                
                if st.button("Upload All Signatures"):
                    results = []
                    for idx, signature in enumerate(processed_signatures):
                        success, message = upload_signature(selected_user, signature)
                        if success:
                            results.append(f"Signature {idx+1}: Uploaded successfully")
                        else:
                            results.append(f"Signature {idx+1}: {message}")
                    st.write("\n".join(results))
def display_performance_page():
    if not st.session_state.is_admin:
        st.error("Admin access required")
        return
    
    st.header("System Performance")
    
    # Show latest operation time
    if st.session_state.last_operation_time is not None:
        st.metric("Last Operation Time", f"{st.session_state.last_operation_time:.4f} seconds")
    
    # System info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Information")
        st.write(f"Python version: {sys.version}")
        st.write(f"OpenCV version: {cv2.__version__}")
        st.write(f"Current time: {datetime.datetime.now()}")
    
    with col2:
        st.subheader("Storage Usage")
        # Get total storage used
        total_size = 0
        try:
            blobs = list(storage_client.list_blobs(signature_bucket.name))
            for blob in blobs:
                total_size += blob.size
            st.write(f"Total storage used: {total_size / (1024*1024):.2f} MB")
            st.write(f"Total objects: {len(blobs)}")
        except Exception as e:
            st.error(f"Error fetching storage info: {str(e)}")
    
    # Performance metrics
    st.subheader("Operation Performance")
    if not st.session_state.operation_durations:
        st.info("No operations recorded yet")
    else:
        # Convert to DataFrame for plotting
        perf_data = pd.DataFrame(st.session_state.operation_durations)
        
        # Group by operation and calculate stats
        operation_stats = perf_data.groupby('operation')['duration'].agg(['mean', 'min', 'max', 'count']).reset_index()
        operation_stats = operation_stats.sort_values(by='mean', ascending=False)
        
        # Rename columns for display
        operation_stats.columns = ['Operation', 'Avg Time (s)', 'Min Time (s)', 'Max Time (s)', 'Count']
        
        # Round for better display
        for col in ['Avg Time (s)', 'Min Time (s)', 'Max Time (s)']:
            operation_stats[col] = operation_stats[col].round(4)
        
        st.dataframe(operation_stats, use_container_width=True)
        
        # Plot performance over time
        st.subheader("Performance Over Time")
        try:
            # Sort by timestamp and take last 50 operations
            plot_data = perf_data.sort_values('timestamp').tail(50)
            
            # Create a pivot table for plotting multiple operations
            fig, ax = plt.subplots(figsize=(10, 5))
            for op in plot_data['operation'].unique():
                op_data = plot_data[plot_data['operation'] == op]
                ax.plot(op_data['timestamp'], op_data['duration'], marker='o', label=op)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title('Operation Performance Over Time')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating performance chart: {str(e)}")
    
    # Add system monitoring and tuning options
    with st.expander("System Tuning"):
        st.subheader("Cache Management")
        if st.button("Clear Cache"):
            # Clear Streamlit cache
            st.cache_data.clear()
            st.success("Cache cleared successfully")
        
        # Performance tuning sliders
        st.subheader("Performance Parameters")
        new_orb_features = st.slider("ORB Features", min_value=100, max_value=1000, value=500, step=100)
        
        if st.button("Apply Settings"):
            # Create a new global ORB detector with updated settings
            global ORB_DETECTOR
            ORB_DETECTOR = cv2.ORB_create(nfeatures=new_orb_features)
            st.success("Settings applied successfully")

if __name__ == "__main__":
    main()