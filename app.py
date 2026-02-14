import cv2
import pickle
import numpy as np
import os
import csv
import time
import sqlite3
import hashlib
import secrets
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    session,
    flash,
    jsonify,
    make_response,
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)


# Database setup
def get_db_connection():
    conn = sqlite3.connect("attendance.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with open("schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


# Initialize database if it doesn't exist
if not os.path.exists("attendance.db"):
    init_db()


def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)


# Initialize Haar Cascade
# First try to use the local file
local_cascade_path = (
    r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\haarcascade_frontalface_default.xml"
)
if os.path.exists(local_cascade_path):
    print(f"Using local cascade file: {local_cascade_path}")
    facedetect = cv2.CascadeClassifier(local_cascade_path)
else:
    # Fallback to OpenCV's built-in cascade file
    print("Local cascade file not found, using OpenCV's built-in cascade")
    facedetect = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# Check if the classifier loaded successfully
if facedetect.empty():
    print("ERROR: Could not load the face cascade classifier!")
    # Try alternate locations
    alt_path = "haarcascade_frontalface_default.xml"  # Try without space
    if os.path.exists(alt_path):
        print(f"Using alternate cascade file: {alt_path}")
        facedetect = cv2.CascadeClassifier(alt_path)

# Create directory for saving data
if not os.path.exists("data"):
    os.makedirs("data")


# Function to capture new faces
def capture_faces(name):
    # Make sure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")

    # Initialize the camera - try DirectShow first, then default
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video.isOpened():
        print("DirectShow backend failed, trying default backend")
        video = cv2.VideoCapture(0)

    # Set basic camera properties
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if camera opened successfully
    if not video.isOpened():
        print("Error: Could not open camera with any backend")
        return "Error: Camera not accessible. Please check your webcam connection."

    # Display status message
    print(f"Starting face capture for {name}")

    faces_data = []
    required_samples = 50  # Number of face captures needed

    # Simpler frame skipping for diversity
    capture_delay = 2  # Capture every 2 frames
    frame_count = 0

    # Simple countdown before starting
    countdown = 3
    countdown_time = time.time()
    capturing_started = False
    capture_timeout = 30  # seconds timeout for face detection
    capture_start_time = None

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        if not ret:
            print("Error accessing camera.")
            break

        # Get current time
        current_time = time.time()

        # Make a copy for display
        display_frame = frame.copy()

        # Show countdown before starting capture
        if not capturing_started:
            # Draw a background for text
            cv2.rectangle(display_frame, (0, 0), (640, 100), (40, 40, 40), -1)

            # Update countdown every second
            if current_time - countdown_time >= 1:
                countdown -= 1
                countdown_time = current_time
                if countdown <= 0:
                    capturing_started = True
                    capture_start_time = current_time

            # Draw countdown message
            cv2.putText(
                display_frame,
                f"Get ready! Starting in {countdown}...",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Show the frame
            cv2.imshow("Face Capture", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        # Process frame for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces - use more reliable parameters
        faces = facedetect.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )

        # Draw simple UI - progress info
        cv2.rectangle(display_frame, (0, 0), (640, 50), (40, 40, 40), -1)
        progress_text = f"Capturing: {len(faces_data)}/{required_samples} images"
        cv2.putText(
            display_frame,
            progress_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Found a face?
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture face every few frames
            if len(faces_data) < required_samples and frame_count % capture_delay == 0:
                # Get face with small padding
                padding = int(0.1 * w)  # 10% padding
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)

                # Crop and resize the face
                try:
                    crop_img = frame[y1:y2, x1:x2]
                    if crop_img.size > 0:  # Make sure we have a valid image
                        resized_img = cv2.resize(crop_img, (50, 50))
                        faces_data.append(resized_img)

                        # Simple visual feedback - green border means captured
                        cv2.rectangle(
                            display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3
                        )
                except Exception as e:
                    print(f"Error cropping face: {e}")
        else:
            # No face detected
            cv2.putText(
                display_frame,
                "No face detected",
                (180, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Increment frame counter
        frame_count += 1

        # Show the frame
        cv2.imshow("Face Capture", display_frame)

        # Abort if no face detected within timeout
        if capture_start_time and time.time() - capture_start_time > capture_timeout:
            print(
                f"Timeout: No face detected within {capture_timeout} seconds, aborting capture."
            )
            video.release()
            cv2.destroyAllWindows()
            return "Error: Could not detect face. Please ensure proper lighting and try again."

        # Check for user quit or completion
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or len(faces_data) >= required_samples:
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

    # Make sure we have enough samples
    if len(faces_data) < required_samples:
        print(
            f"Warning: Only captured {len(faces_data)} samples out of required {required_samples}"
        )
        # If we have at least 1 sample, we can proceed, otherwise return error
        if len(faces_data) < 1:
            return "Error: Could not capture any face samples. Please try again with better lighting."
        # Adjust required_samples to what we have
        required_samples = len(faces_data)

    print(f"Successfully captured {len(faces_data)} face samples")

    # Save the data - ensure consistent dimensions (50x50 = 7500 pixels per face for RGB)
    faces_data = np.asarray(faces_data)
    # Check for empty array
    if len(faces_data) == 0:
        return "Error: No face samples captured. Please try again with better lighting."

    # Print shape before reshaping to debug
    print(f"Face data shape before reshaping: {faces_data.shape}")
    # Make sure all faces have the right dimensions - 50x50 RGB images = 7500 elements per face
    try:
        # First ensure all samples have the same shape by resizing if needed
        for i in range(len(faces_data)):
            if faces_data[i].shape != (50, 50, 3):
                faces_data[i] = cv2.resize(faces_data[i], (50, 50))

        # Then reshape to a 2D array for the classifier
        faces_data = faces_data.reshape(len(faces_data), 50 * 50 * 3)
        print(f"Face data successfully reshaped to: {faces_data.shape}")
    except Exception as e:
        print(f"Error reshaping face data: {str(e)}")
        # Create a fresh array with correct dimensions as fallback
        faces_data = np.zeros((len(faces_data), 50 * 50 * 3))
        for i in range(len(faces_data)):
            flat = cv2.resize(faces_data[i], (50, 50)).flatten()
            faces_data[i] = flat if flat.size == 50 * 50 * 3 else np.zeros(50 * 50 * 3)
        print(f"Created fallback face data with shape: {faces_data.shape}")

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created data directory")

    try:
        if not os.path.exists("data/names.pkl"):
            names = [name] * len(faces_data)
            with open("data/names.pkl", "wb") as f:
                pickle.dump(names, f)
            print("Created new names file")
        else:
            with open("data/names.pkl", "rb") as f:
                names = pickle.load(f)
            names += [name] * len(faces_data)
            with open("data/names.pkl", "wb") as f:
                pickle.dump(names, f)
            print(f"Updated existing names file, now contains {len(names)} samples")

        # Save the faces data
        if not os.path.exists("data/faces_data.pkl"):
            with open("data/faces_data.pkl", "wb") as f:
                pickle.dump(faces_data, f)
            print("Created new faces data file")
        else:
            try:
                with open("data/faces_data.pkl", "rb") as f:
                    faces = pickle.load(f)

                # Print shapes for debugging
                print(f"Existing faces data shape: {faces.shape}")
                print(f"New faces data shape: {faces_data.shape}")

                # Check and fix dimensions if needed
                if faces.size == 0:
                    print("Existing faces data is empty, using new data only")
                    faces = faces_data
                elif faces.shape[1] != faces_data.shape[1]:
                    print(
                        f"Dimension mismatch: existing={faces.shape[1]}, new={faces_data.shape[1]}"
                    )
                    # Try to reshape old data to match new data dimensions
                    try:
                        # If old data has wrong dimensions, we'll recreate it with the right dimensions
                        # We'll store the old data temporarily for names matching
                        with open("data/names.pkl", "rb") as f:
                            old_names = pickle.load(f)

                        # Create new faces array with right dimensions
                        print("Recreating faces data with consistent dimensions")
                        # Just use the new data and discard old data with wrong dimensions
                        # This is safer than trying to reshape incompatible data
                        faces = faces_data

                        # Update names list to match the new face data size
                        # This ensures names and faces stay aligned
                        with open("data/names.pkl", "wb") as f:
                            pickle.dump([name] * len(faces_data), f)
                        print("Names file has been reset to match new face data")
                    except Exception as e:
                        print(f"Error fixing dimension mismatch: {str(e)}")
                        faces = faces_data
                else:
                    # Append only if dimensions match
                    faces = np.append(faces, faces_data, axis=0)
                    print(
                        f"Successfully appended new face data, new shape: {faces.shape}"
                    )
            except Exception as e:
                print(
                    f"Error loading existing face data: {str(e)}. Creating new dataset."
                )
                faces = faces_data
            with open("data/faces_data.pkl", "wb") as f:
                pickle.dump(faces, f)
            print(f"Updated faces data file, now contains {faces.shape[0]} samples")
    except Exception as e:
        print(f"Error saving face data: {str(e)}")
        return f"Error saving data: {str(e)}"

    return "Dataset created successfully!"


# Function to start face recognition and attendance
def start_recognition(class_id=None):
    try:
        # Check if data directory exists
        if not os.path.exists("data"):
            print("Error: Data directory not found")
            os.makedirs("data")
            print("Created data directory, but no facial data exists yet")
            return "Please register at least one face before starting recognition"

        # Check if required files exist with detailed feedback
        missing_files = []
        if not os.path.exists("data/names.pkl"):
            missing_files.append("names.pkl")
        if not os.path.exists("data/faces_data.pkl"):
            missing_files.append("faces_data.pkl")

        if missing_files:
            print(f"Error: Required data files not found: {', '.join(missing_files)}")
            return "Please register at least one face before starting recognition"

        # Load Data for Recognition
        try:
            with open("data/names.pkl", "rb") as f:
                names = pickle.load(f)
            print(f"Successfully loaded names.pkl with {len(names)} names")
            if len(names) == 0:
                print("Error: names list is empty")
                return "No registered users found. Please register at least one face before starting recognition"
        except Exception as e:
            print(f"Error loading names.pkl: {str(e)}")
            return "Error loading face data. Please try registering your face again"

        try:
            with open("data/faces_data.pkl", "rb") as f:
                faces_data = pickle.load(f)
            print(f"Successfully loaded faces_data.pkl with shape {faces_data.shape}")
            if faces_data.size == 0:
                print("Error: faces_data array is empty")
                return "Error: No face data found"
        except Exception as e:
            print(f"Error loading faces_data.pkl: {str(e)}")
            return "Error: Failed to load faces data"

        try:
            faces_data = faces_data.reshape(faces_data.shape[0], -1)
            print(f"Successfully reshaped faces_data to {faces_data.shape}")
            print(f"Number of face samples: {faces_data.shape[0]}")
            print(f"Number of names: {len(names)}")

            # Fix inconsistency between faces and names instead of returning error
            if faces_data.shape[0] != len(names):
                print(
                    f"Warning: Mismatch between faces_data.shape[0] ({faces_data.shape[0]}) and len(names) ({len(names)})"
                )
                print("Attempting to fix the inconsistency automatically...")

                try:
                    # Case 1: More faces than names - truncate face data to match names
                    if faces_data.shape[0] > len(names):
                        print("More faces than names, truncating face data")
                        faces_data = faces_data[: len(names)]
                    # Case 2: More names than faces - duplicate the last name for each remaining face
                    else:
                        print("More names than faces, adjusting names list")
                        # Take a subset of names to match number of faces
                        names = names[: faces_data.shape[0]]

                    # Save the corrected data
                    with open("data/names.pkl", "wb") as f:
                        pickle.dump(names, f)
                    with open("data/faces_data.pkl", "wb") as f:
                        pickle.dump(faces_data, f)
                    print("Successfully fixed data inconsistency")
                except Exception as e:
                    print(f"Error fixing data inconsistency: {str(e)}")
                    return "Error: Failed to fix inconsistent face data. Please register your face again."
        except Exception as e:
            print(f"Error reshaping faces_data: {str(e)}")
            return "Error: Failed to process faces data"

        # Initialize KNN Classifier with robust error handling
        try:
            if faces_data.shape[0] < 1:
                print("Error: No face samples for KNN classifier")
                return "Error: No face samples for recognition"

            # Make sure names and faces_data have same length
            if len(names) != faces_data.shape[0]:
                print("Warning: Length mismatch before training, fixing...")
                # Use the minimum length between names and faces_data
                min_len = min(len(names), faces_data.shape[0])
                names = names[:min_len]
                faces_data = faces_data[:min_len]

            # Use a more appropriate number of neighbors
            n_neighbors = min(3, faces_data.shape[0])
            print(f"Using {n_neighbors} neighbors for KNN classification")

            # Verify data is suitable for training (no NaNs or infs)
            if np.isnan(faces_data).any() or np.isinf(faces_data).any():
                print("Warning: Data contains NaN or Inf values, cleaning data...")
                # Replace NaN and Inf values with zeros
                faces_data = np.nan_to_num(faces_data)

            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(faces_data, names)
            print("Successfully trained KNN classifier")
        except Exception as e:
            print(f"Error training classifier: {str(e)}")
            return f"Error: Could not train recognition model: {str(e)}"

        # Video Capture for Recognition - use the same settings as in capture
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not video.isOpened():
            print("DirectShow backend failed, trying default backend")
            video = cv2.VideoCapture(0)

        # Set basic camera properties to match capture function
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not video.isOpened():
            print("Error: Could not open video capture device with any backend")
            return "Error: Camera not accessible"

        # Create a simple colored background instead of loading an image
        # This eliminates any dependency on external image files
        imgBackground = np.zeros((480, 640, 3), dtype=np.uint8)
        # Fill with a pleasant blue gradient background
        for y in range(480):
            for x in range(640):
                # Create a gradient from dark blue to light blue
                blue_value = int(180 + (y / 480) * 75)  # 180-255 range for blue
                imgBackground[y, x] = (100, 50, blue_value)  # BGR format

        print("Using generated background instead of image file")

        COL_NAMES = ["NAME", "DATE", "TIME"]

        attendance_dir = r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\Attendance"
        os.makedirs(attendance_dir, exist_ok=True)

        # Instead of using a fixed recognition threshold, we'll use confidence level
        min_confidence_for_attendance = (
            70  # Minimum confidence percentage to record attendance
        )

        attendance_list = []  # List to store multiple attendance records
        attendance_recorded = (
            {}
        )  # Dictionary to track when attendance was last recorded

        while True:
            ret, frame = video.read()
            if not ret:
                print("Error accessing camera.")
                break

            # Process frame with improved parameters for better recognition
            # Use the full frame for better quality
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization to improve contrast
            gray = cv2.equalizeHist(gray)

            # Use more lenient detection parameters
            faces = facedetect.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More sensitive scale factor
                minNeighbors=3,  # Balanced for accuracy and detection ease
                minSize=(20, 20),  # Minimum face size - same as in capture
            )

            # Display guidance text with instructions
            cv2.rectangle(frame, (0, 0), (640, 80), (40, 40, 40), -1)
            cv2.putText(
                frame,
                "Position your face in front of the camera",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Window will close automatically after attendance is recorded",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit without recording, 'm' for manual attendance",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            for x, y, w, h in faces:
                crop_img = frame[y : y + h, x : x + w, :]
                # Make sure we have a valid image
                if crop_img.size == 0 or crop_img is None:
                    continue

                # Apply same preprocessing as during capture
                try:
                    resized_img = cv2.resize(crop_img, (50, 50))
                    flattened_img = resized_img.flatten().reshape(1, -1)

                    # Normalize the image for better recognition
                    norm_img = cv2.normalize(
                        flattened_img, None, 0, 255, cv2.NORM_MINMAX
                    )
                except Exception as e:
                    print(f"Error processing face image: {e}")
                    continue

                # Try to predict the name in real-time
                try:
                    # Get prediction directly - use voting from multiple neighbors
                    prediction = knn.predict(norm_img)[0]

                    # Get probabilities or distance from neighbors
                    n_neighbors = min(5, faces_data.shape[0])
                    distances, indices = knn.kneighbors(
                        norm_img, n_neighbors=n_neighbors
                    )

                    # Calculate confidence based on neighbor votes instead of distance
                    neighbors = [names[idx] for idx in indices[0]]
                    votes = {}
                    for neighbor in neighbors:
                        if neighbor in votes:
                            votes[neighbor] += 1
                        else:
                            votes[neighbor] = 1

                    # Get vote count for the predicted name
                    vote_count = votes.get(prediction, 0)
                    confidence = (vote_count / n_neighbors) * 100

                    # Print recognition info
                    print(
                        f"Predicted: {prediction}, Confidence: {confidence:.0f}%, Votes: {votes}"
                    )

                    # Determine if face is recognized with sufficient confidence
                    if confidence < 40:  # Very low confidence
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown
                    else:
                        # Get the predicted name with confidence
                        name = prediction
                        name = f"{name} ({confidence:.0f}%)"
                        color = (0, 255, 0)  # Green for recognized

                        # Automatically record attendance if confidence is high enough
                        if confidence >= min_confidence_for_attendance:
                            # Check if this person's attendance was already recorded recently
                            current_time = time.time()
                            # Only record if we haven't recorded in the last 60 seconds
                            if (
                                prediction not in attendance_recorded
                                or (current_time - attendance_recorded[prediction]) > 60
                            ):
                                ts = time.time()
                                record_date = datetime.fromtimestamp(ts).strftime(
                                    "%d-%m-%Y"
                                )
                                timestamp = datetime.fromtimestamp(ts).strftime(
                                    "%H:%M:%S"
                                )
                                attendance_list.append(
                                    [prediction, record_date, timestamp]
                                )
                                speak(f"Attendance recorded for {prediction}")
                                attendance_recorded[prediction] = current_time

                                # Show confirmation message
                                cv2.rectangle(
                                    frame, (0, 0), (640, 100), (0, 150, 0), -1
                                )
                                cv2.putText(
                                    frame,
                                    f"Attendance recorded for {prediction}",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                                cv2.putText(
                                    frame,
                                    "Saving and closing in 2 seconds...",
                                    (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    1,
                                )
                                cv2.imshow("Face Recognition", frame)
                                cv2.waitKey(2000)  # Show confirmation for 2 seconds

                                # Save attendance immediately
                                save_attendance_now(
                                    attendance_list, attendance_dir, COL_NAMES, class_id
                                )

                                # Close video and window
                                video.release()
                                cv2.destroyAllWindows()
                                return "Attendance recorded successfully!"
                except Exception as e:
                    name = "Error"
                    color = (0, 165, 255)  # Orange for error
                    print(f"Error during prediction: {str(e)}")

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Create dark background for text
                cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)

                # Display name on rectangle
                cv2.putText(
                    frame,
                    name,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1)
            if key == ord("o"):  # Press 'o' to mark attendance
                for x, y, w, h in faces:
                    crop_img = frame[y : y + h, x : x + w, :]
                    resized_img = (
                        cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    )

                    # Apply preprocessing to the face image for better recognition
                    # Normalize the image to reduce lighting effects
                    norm_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX)

                    # Try to predict with error handling - simpler approach for manual attendance
                    try:
                        # Just get the prediction directly
                        prediction = knn.predict(norm_img)[0]
                        name = prediction
                        color = (0, 255, 0)  # Green for recognized
                    except Exception as e:
                        print(f"Error during prediction: {str(e)}")
                        name = "Error"
                        color = (0, 165, 255)  # Orange for error

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(
                        frame,
                        str(name),
                        (x, y - 15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1,
                    )

                    ts = time.time()
                    record_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    attendance_list.append([name, record_date, timestamp])
                    speak(f"Attendance recorded for {name}")

            elif key == ord("m"):  # Press 'm' to manually record attendance
                for x, y, w, h in faces:
                    crop_img = frame[y : y + h, x : x + w, :]
                    if crop_img.size == 0 or crop_img is None:
                        continue

                    try:
                        resized_img = cv2.resize(crop_img, (50, 50))
                        flattened_img = resized_img.flatten().reshape(1, -1)
                        norm_img = cv2.normalize(
                            flattened_img, None, 0, 255, cv2.NORM_MINMAX
                        )

                        # Get prediction
                        prediction = knn.predict(norm_img)[0]

                        # Manual recording always uses the predicted name
                        ts = time.time()
                        record_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        attendance_list.append([prediction, record_date, timestamp])
                        speak(f"Attendance manually recorded for {prediction}")

                        # Update screen with confirmation
                        cv2.rectangle(frame, (0, 0), (640, 40), (0, 150, 0), -1)
                        cv2.putText(
                            frame,
                            f"Attendance recorded for {prediction}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(1000)  # Show the confirmation for 1 second
                    except Exception as e:
                        print(f"Error recording manual attendance: {e}")

            elif key == ord("q"):  # Press 'q' to quit
                break

        # If we reach here, the user manually quit without recording attendance
        video.release()
        cv2.destroyAllWindows()

        # Save attendance if any was recorded but not saved
        if attendance_list:
            save_attendance_now(attendance_list, attendance_dir, COL_NAMES)

        return "Recognition complete and attendance taken!"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: An unexpected error occurred - {str(e)}"


# Function to save attendance immediately
def save_attendance_now(attendance_list, attendance_dir, COL_NAMES, class_id=None):
    if not attendance_list:
        return

    # Create attendance file with current date
    current_date = datetime.now().strftime("%d-%m-%Y")
    attendance_file = os.path.join(attendance_dir, f"Attendance_{current_date}.csv")
    attendance_exists = os.path.exists(attendance_file)

    # Write all collected attendance records
    with open(attendance_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not attendance_exists:
            writer.writerow(COL_NAMES)
        # Write all records
        writer.writerows(attendance_list)

    # Store attendance in database regardless of teacher login
    for record in attendance_list:
        student_name, date_str, time_str = record
        if student_name != "Unknown" and student_name != "Error":
            # Pass the class_id to the store_attendance_in_db function
            store_attendance_in_db(student_name, date_str, time_str, class_id)
    speak("Attendance has been saved")
    return "Attendance saved successfully!"


# User Management functions
def get_all_users():
    if os.path.exists("data/names.pkl"):
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        return list(set(names))  # Return unique names
    return []


def delete_user(username):
    if os.path.exists("data/names.pkl"):
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        # Remove all instances of the username
        names = [name for name in names if name != username]
        with open("data/names.pkl", "wb") as f:
            pickle.dump(names, f)
        return True
    return False


# Settings functions
def save_settings(settings_dict):
    with open("data/settings.pkl", "wb") as f:
        pickle.dump(settings_dict, f)


def load_settings():
    if os.path.exists("data/settings.pkl"):
        with open("data/settings.pkl", "rb") as f:
            return pickle.load(f)
    return {"camera_index": 0, "required_samples": 50, "attendance_threshold": 0.6}


# Export functions
def export_attendance_csv():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_export_{timestamp}.csv"
    export_path = os.path.join("data", filename)

    if os.path.exists("data/attendance.csv"):
        with open("data/attendance.csv", "r") as source:
            with open(export_path, "w", newline="") as target:
                reader = csv.reader(source)
                writer = csv.writer(target)
                for row in reader:
                    writer.writerow(row)
        return filename
    return None


# Routes
@app.route("/")
def home():
    return render_template("index.html", teacher_logged_in="teacher_id" in session)


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")


@app.route("/capture", methods=["GET", "POST"])
def capture():
    if request.method == "POST":
        name = request.form["name"]
        # Check if name contains only letters and spaces
        if (
            name
            and all(c.isalpha() or c.isspace() for c in name)
            and not name.isspace()
        ):
            result = capture_faces(name)
            return render_template("result.html", result=result)
        else:
            return render_template(
                "capture.html",
                error="Please enter a valid name using only alphabets and spaces between names.",
            )
    return render_template("capture.html")


@app.route("/api/upload_capture", methods=["POST"])
def upload_capture():
    """
    API endpoint for web-based face capture.
    Receives base64 encoded images and saves them for training.
    """
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        user_id = data.get("userId", "").strip()
        department = data.get("department", "").strip()
        phone = data.get("phone", "").strip()
        role = data.get("role", "").strip()
        notes = data.get("notes", "").strip()
        images = data.get("images", [])
        
        # Validate input
        if not username or not images:
            return jsonify({"error": "Invalid input"}), 400
        
        if not all(c.isalpha() or c.isspace() for c in username):
            return jsonify({"error": "Invalid username"}), 400
        
        if not email or '@' not in email:
            return jsonify({"error": "Invalid email address"}), 400
        
        if not user_id:
            return jsonify({"error": "ID number is required"}), 400
        
        if not department:
            return jsonify({"error": "Department is required"}), 400
        
        if not role:
            return jsonify({"error": "Role is required"}), 400
        
        # Require at least 20 images for good accuracy
        if len(images) < 20:
            return jsonify({"error": "At least 20 images required for accurate recognition"}), 400
        
        # Limit to prevent processing too many images
        if len(images) > 100:
            images = images[:100]  # Use first 100 if more provided
        
        # Create data directory
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Process and save images
        faces_data = []
        
        for idx, image_data in enumerate(images):
            try:
                # Remove data:image/jpeg;base64, prefix if present
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                
                # Decode base64
                import base64
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                    # Crop and resize
                    crop_img = frame[y:y+h, x:x+w]
                    resized_img = cv2.resize(crop_img, (50, 50))
                    faces_data.append(resized_img)
            
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        if len(faces_data) < 10:
            return jsonify({"error": "Could not detect faces in images"}), 400
        
        # Convert to numpy array
        faces_data = np.array(faces_data)
        
        # Load existing data if present
        try:
            if os.path.exists("data/faces_data.pkl"):
                with open("data/faces_data.pkl", "rb") as f:
                    existing_faces = pickle.load(f)
                faces_data = np.append(existing_faces, faces_data, axis=0)
            
            if os.path.exists("data/names.pkl"):
                with open("data/names.pkl", "rb") as f:
                    existing_names = pickle.load(f)
                new_names = [username] * len(faces_data)
                names = existing_names + new_names[len(existing_names):]
            else:
                names = [username] * len(faces_data)
        
        except Exception as e:
            print(f"Error loading existing data: {e}")
            names = [username] * len(faces_data)
        
        # Save updated data
        with open("data/faces_data.pkl", "wb") as f:
            pickle.dump(faces_data, f)
        
        with open("data/names.pkl", "wb") as f:
            pickle.dump(names, f)
        
        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces_data.reshape(faces_data.shape[0], -1), names)
        
        with open("data/face_recognizer.pkl", "wb") as f:
            pickle.dump(knn, f)
        
        # Add user to database with all details
        conn = get_db_connection()
        try:
            # First create the users table if it doesn't exist with all fields
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    user_id TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL,
                    phone TEXT,
                    role TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Insert or replace user data
            conn.execute(
                '''INSERT OR REPLACE INTO users 
                   (username, name, email, user_id, department, phone, role, notes, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (username, username, email, user_id, department, phone, role, notes, datetime.now().isoformat())
            )
            conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()
        
        return jsonify({
            "success": True,
            "message": f"Successfully registered {username} with {len(faces_data)} face samples",
            "userData": {
                "name": username,
                "email": email,
                "userId": user_id,
                "department": department,
                "role": role
            }
        }), 200
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    # Check if a teacher is logged in
    teacher_logged_in = "teacher_id" in session
    classes = []

    # Get classes for the teacher if logged in
    if teacher_logged_in:
        conn = get_db_connection()
        classes = conn.execute(
            "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
        conn.close()

    if request.method == "POST":
        # Get the selected class ID if provided
        class_id = request.form.get("class_id")
        if class_id:
            class_id = int(class_id)

        # Start recognition with the selected class
        result = start_recognition(class_id)
        return render_template("result.html", result=result)

    return render_template(
        "recognize.html", teacher_logged_in=teacher_logged_in, classes=classes
    )


@app.route("/attendance")
def show_attendance():
    # Get filter parameters from request
    # Convert date format: HTML input uses YYYY-MM-DD, DB uses DD-MM-YYYY
    filter_date_input = request.args.get('date', '')
    
    # If no date provided, default to today in DD-MM-YYYY format
    if not filter_date_input:
        filter_date = datetime.now().strftime("%d-%m-%Y")
        filter_date_html = datetime.now().strftime("%Y-%m-%d")
    else:
        # Convert from YYYY-MM-DD to DD-MM-YYYY for database query
        try:
            date_obj = datetime.strptime(filter_date_input, "%Y-%m-%d")
            filter_date = date_obj.strftime("%d-%m-%Y")
            filter_date_html = filter_date_input
        except:
            filter_date = datetime.now().strftime("%d-%m-%Y")
            filter_date_html = datetime.now().strftime("%Y-%m-%d")
    
    filter_department = request.args.get('department', '')
    filter_name = request.args.get('name', '')
    
    # Initialize empty list for attendance records
    attendance_records = []
    departments = []
    
    try:
        conn = get_db_connection()
        
        # Get all departments for filter dropdown
        all_depts = conn.execute(
            "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != ''"
        ).fetchall()
        departments = [d['department'] for d in all_depts]
        
        # Build query with filters - use LIKE for better matching
        query = """
            SELECT 
                ar.student_name,
                ar.date,
                ar.time,
                ar.status,
                u.user_id,
                u.department,
                u.email
            FROM attendance_records ar
            LEFT JOIN users u ON (
                u.name LIKE '%' || ar.student_name || '%' OR 
                ar.student_name LIKE '%' || u.name || '%' OR
                ar.student_name = u.username
            )
            WHERE ar.student_name != "Unknown" AND ar.student_name != "Error"
        """
        
        params = []
        
        # Apply date filter (default to today)
        if filter_date:
            query += " AND ar.date = ?"
            params.append(filter_date)
        
        # Apply department filter
        if filter_department:
            query += " AND u.department = ?"
            params.append(filter_department)
        
        # Apply name filter
        if filter_name:
            query += " AND (ar.student_name LIKE ? OR u.user_id LIKE ?)"
            params.append(f"%{filter_name}%")
            params.append(f"%{filter_name}%")
        
        query += " ORDER BY ar.date DESC, ar.time DESC"
        
        db_records = conn.execute(query, params).fetchall()
        
        # Convert database records to list format
        for record in db_records:
            attendance_records.append({
                'name': record['student_name'],
                'date': record['date'],
                'time': record['time'],
                'status': record['status'] or 'Present',
                'user_id': record['user_id'] or 'N/A',
                'department': record['department'] or 'N/A',
                'email': record['email'] or 'N/A'
            })
        
        # Calculate statistics
        today = datetime.now().strftime("%d-%m-%Y")
        total_records = len(db_records)
        unique_users = len(set(r['student_name'] for r in db_records))
        today_count = sum(1 for r in db_records if r['date'] == today)
        
        # Department-wise stats for filtered results
        dept_stats = {}
        for record in db_records:
            dept = record['department'] or 'N/A'
            if dept not in dept_stats:
                dept_stats[dept] = 0
            dept_stats[dept] += 1
        
        conn.close()
        
    except Exception as e:
        print(f"Error querying database for attendance records: {str(e)}")
        import traceback
        traceback.print_exc()
        attendance_records = []
        departments = []
        total_records = 0
        unique_users = 0
        today_count = 0
        dept_stats = {}
    
    has_records = len(attendance_records) > 0
    
    stats = {
        'total_records': total_records,
        'unique_users': unique_users,
        'today_count': today_count,
        'dept_stats': dept_stats
    }

    return render_template(
        "attendance.html", 
        attendance_data=attendance_records, 
        has_records=has_records,
        stats=stats,
        departments=departments,
        filter_date=filter_date_html,
        filter_department=filter_department,
        filter_name=filter_name,
        current_date=datetime.now().strftime("%d-%m-%Y")
    )


@app.route("/manage_users")
def manage_users():
    # Get all users from names.pkl
    users = get_all_users()
    
    # Also get detailed user info from database
    user_details = []
    conn = get_db_connection()
    for username in users:
        user_info = conn.execute(
            "SELECT name, email, user_id, department, role FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if user_info:
            user_details.append({
                'username': username,
                'name': user_info['name'] if user_info['name'] else username,
                'email': user_info['email'],
                'user_id': user_info['user_id'],
                'department': user_info['department'],
                'role': user_info['role']
            })
        else:
            user_details.append({
                'username': username,
                'name': username,
                'email': None,
                'user_id': None,
                'department': None,
                'role': None
            })
    conn.close()
    
    return render_template("manage_users.html", users=users, user_details=user_details)


@app.route("/delete_user/<username>")
def delete_user_route(username):
    if delete_user(username):
        flash(f"User '{username}' has been deleted successfully.", "success")
        return redirect(url_for("manage_users"))
    flash(f"Error deleting user '{username}'.", "danger")
    return redirect(url_for("manage_users"))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        new_settings = {
            "camera_index": int(request.form.get("camera_index", 0)),
            "required_samples": int(request.form.get("required_samples", 50)),
            "attendance_threshold": float(
                request.form.get("attendance_threshold", 0.6)
            ),
        }
        save_settings(new_settings)
        flash("Settings saved successfully!", "success")
        return redirect(url_for("settings"))

    current_settings = load_settings()
    return render_template("settings.html", settings=current_settings)


# API Routes for Real-time Face Recognition
@app.route("/api/capture_frame", methods=["POST"])
def capture_frame():
    """Receive and process a captured frame from the client"""
    try:
        import base64
        import numpy as np
        from io import BytesIO
        
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'success': False, 'message': 'No frame data received'})
        
        # Decode the base64 image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Failed to decode frame'})
        
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        faces_detected = len(faces)
        recognition_results = []
        
        if faces_detected > 0 and os.path.exists("data/names.pkl") and os.path.exists("data/faces_data.pkl"):
            try:
                # Load face data
                with open("data/names.pkl", "rb") as f:
                    names = pickle.load(f)
                with open("data/faces_data.pkl", "rb") as f:
                    faces_data = pickle.load(f)
                
                faces_data = faces_data.reshape(faces_data.shape[0], -1)
                
                # Train classifier
                n_neighbors = min(3, faces_data.shape[0])
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(faces_data, names)
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y+h, x:x+w]
                    crop_img_resized = cv2.resize(crop_img, (50, 50))
                    crop_img_resized_flat = crop_img_resized.reshape(1, -1)
                    
                    # Get prediction
                    output = knn.predict(crop_img_resized_flat)
                    confidence = knn.predict_proba(crop_img_resized_flat)
                    confidence_score = max(confidence[0]) * 100
                    
                    recognition_results.append({
                        'name': str(output[0]),
                        'confidence': round(confidence_score, 2),
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h)
                    })
            except Exception as e:
                print(f"Error in recognition: {str(e)}")
        
        return jsonify({
            'success': True,
            'faces_detected': faces_detected,
            'results': recognition_results
        })
    
    except Exception as e:
        print(f"Error in capture_frame: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route("/api/save_attendance", methods=["POST"])
def save_attendance_api():
    """Save recognized attendance to database and CSV - only once per student per day"""
    try:
        data = request.get_json()
        name = data.get('name')
        class_id = data.get('class_id')
        teacher_id = data.get('teacher_id', 1)  # Default teacher_id if not provided
        
        if not name:
            return jsonify({'success': False, 'message': 'No name provided'})
        
        conn = get_db_connection()
        
        # Save to database
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if this student already has an attendance record for TODAY (entire day check)
        try:
            existing_record = conn.execute(
                "SELECT * FROM attendance_records WHERE student_name = ? AND date = ?",
                (name, date_str)
            ).fetchone()
            
            if existing_record:
                # Student already recorded today - don't record again
                conn.close()
                return jsonify({
                    'success': False, 
                    'message': f'✓ {name} already recorded for today at {existing_record["time"]}',
                    'duplicate': True,
                    'already_recorded': True
                })
        except Exception as e:
            print(f"Error checking existing records: {str(e)}")
        
        # Record the attendance (first time for this student today)
        try:
            conn.execute(
                "INSERT INTO attendance_records (student_name, date, time, status, class_id, teacher_id) VALUES (?, ?, ?, ?, ?, ?)",
                (name, date_str, time_str, "Present", class_id if class_id else None, teacher_id)
            )
            conn.commit()
            print(f"✓ Attendance recorded for {name} at {time_str}")
        except sqlite3.IntegrityError as ie:
            # Fallback: Attendance already recorded for this student today
            conn.close()
            return jsonify({
                'success': False, 
                'message': f'Attendance already recorded for {name} today',
                'duplicate': True,
                'already_recorded': True
            })
        
        # Save to CSV
        attendance_dir = r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\Attendance"
        os.makedirs(attendance_dir, exist_ok=True)
        
        csv_file = os.path.join(attendance_dir, f"Attendance_{date_str}.csv")
        
        file_exists = os.path.isfile(csv_file)
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["NAME", "DATE", "TIME"])
                writer.writerow([name, date_str, time_str])
            print(f"✓ Attendance saved to CSV for {name}")
        except Exception as csv_error:
            print(f"Error saving to CSV: {str(csv_error)}")
        
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'✓ Attendance recorded for {name}',
            'data': {
                'name': name,
                'date': date_str,
                'time': time_str
            }
        })
    
    except Exception as e:
        print(f"Error in save_attendance_api: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route("/api/get_recognized_users", methods=["GET"])
def get_recognized_users():
    """Get list of all registered users for dropdown"""
    try:
        if not os.path.exists("data/names.pkl"):
            return jsonify({'success': False, 'users': []})
        
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)
        
        return jsonify({
            'success': True,
            'users': list(set(names))  # Remove duplicates
        })
    
    except Exception as e:
        print(f"Error in get_recognized_users: {str(e)}")
        return jsonify({'success': False, 'users': []})


@app.route("/export_attendance", methods=["GET", "POST"])
def export_attendance():
    if request.method == "POST":
        # Handle export request
        format_type = request.form.get('format', 'csv')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        include_headers = request.form.get('include_headers') == 'on'
        include_stats = request.form.get('include_stats') == 'on'
        
        conn = get_db_connection()
        
        # Build query
        query = "SELECT * FROM attendance_records WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        records = conn.execute(query, params).fetchall()
        conn.close()
        
        if format_type == 'csv':
            import io
            output = io.StringIO()
            
            if include_headers:
                output.write("Date,User ID,Status,Time\n")
            
            for record in records:
                output.write(f"{record['date']},{record['user_id']},{record['status']},{record['time']}\n")
            
            if include_stats:
                output.write("\n\nStatistics\n")
                output.write(f"Total Records,{len(records)}\n")
            
            response = make_response(output.getvalue())
            response.headers["Content-Disposition"] = f"attachment;filename=attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            response.headers["Content-Type"] = "text/csv"
            return response
        
        elif format_type == 'json':
            import json
            data = {
                'records': [dict(record) for record in records]
            }
            if include_stats:
                data['statistics'] = {
                    'total_records': len(records),
                    'export_date': datetime.now().isoformat()
                }
            
            response = make_response(json.dumps(data, indent=2))
            response.headers["Content-Disposition"] = f"attachment;filename=attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            response.headers["Content-Type"] = "application/json"
            return response
        
        # For other formats, just return CSV for now
        flash("Export format not fully implemented yet. Using CSV.", "info")
        return redirect(url_for("export_attendance"))
    
    # GET request - show the export form
    conn = get_db_connection()
    total_records = conn.execute("SELECT COUNT(*) as count FROM attendance_records").fetchone()['count']
    conn.close()
    
    stats = {
        'total_records': total_records,
        'total_users': len(get_all_users())
    }
    
    return render_template("export_attendance.html", stats=stats)


# Teacher authentication routes
@app.route("/auth/register", methods=["GET", "POST"])
def teacher_register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        full_name = request.form["full_name"]

        # Hash the password
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO teachers (username, password_hash, email, full_name) VALUES (?, ?, ?, ?)",
                (username, password_hash, email, full_name),
            )
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("teacher_login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()

    return render_template("auth/register.html")


@app.route("/auth/login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Hash the password for comparison
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        teacher = conn.execute(
            "SELECT * FROM teachers WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if teacher and teacher["password_hash"] == password_hash:
            # Create session
            session["teacher_id"] = teacher["id"]
            session["teacher_name"] = teacher["full_name"]
            session["teacher_username"] = teacher["username"]

            flash(f'Welcome back, {teacher["full_name"]}!', "success")
            return redirect(url_for("teacher_dashboard"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("auth/login.html")


@app.route("/auth/logout")
def teacher_logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# Teacher dashboard routes
@app.route("/teacher/dashboard")
def teacher_dashboard():
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get classes for this teacher
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (teacher_id,)
    ).fetchall()

    # Get total students count
    students_count = conn.execute(
        """SELECT COUNT(DISTINCT cs.student_name) as count
           FROM class_students cs
           JOIN classes c ON cs.class_id = c.id
           WHERE c.teacher_id = ?""",
        (teacher_id,)
    ).fetchone()['count']

    # Get recent attendance records (today's records)
    today = datetime.now().strftime("%d-%m-%Y")
    recent_attendance = conn.execute(
        """SELECT ar.*, c.name as class_name 
           FROM attendance_records ar 
           LEFT JOIN classes c ON ar.class_id = c.id 
           WHERE ar.teacher_id = ? AND ar.date = ?
           ORDER BY ar.time DESC LIMIT 10""",
        (teacher_id, today),
    ).fetchall()

    conn.close()

    return render_template(
        "teacher/dashboard.html", 
        classes=classes, 
        recent_attendance=recent_attendance,
        students_count=students_count
    )


# API endpoint to get student count
@app.route("/api/teacher/stats")
def teacher_stats():
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get total students
    students_count = conn.execute(
        """SELECT COUNT(DISTINCT cs.student_name) as count
           FROM class_students cs
           JOIN classes c ON cs.class_id = c.id
           WHERE c.teacher_id = ?""",
        (teacher_id,)
    ).fetchone()['count']

    # Get today's attendance count
    today = datetime.now().strftime("%d-%m-%Y")
    today_attendance = conn.execute(
        """SELECT COUNT(*) as count
           FROM attendance_records
           WHERE teacher_id = ? AND date = ?""",
        (teacher_id, today)
    ).fetchone()['count']

    # Calculate attendance rate
    attendance_rate = 0
    if students_count > 0:
        attendance_rate = round((today_attendance / students_count) * 100, 1)

    conn.close()

    return jsonify({
        "success": True,
        "students_count": students_count,
        "today_attendance": today_attendance,
        "attendance_rate": attendance_rate
    })


# Class management routes
@app.route("/teacher/classes", methods=["GET", "POST"])
def manage_classes():
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    if request.method == "POST":
        name = request.form["name"]
        description = request.form.get("description", "")
        department = request.form.get("department", "")
        teacher_id = session["teacher_id"]

        conn = get_db_connection()
        
        # Check if department column exists, if not add it
        try:
            conn.execute("SELECT department FROM classes LIMIT 1")
        except:
            conn.execute("ALTER TABLE classes ADD COLUMN department TEXT")
            conn.commit()
        
        conn.execute(
            "INSERT INTO classes (teacher_id, name, description, department) VALUES (?, ?, ?, ?)",
            (teacher_id, name, description, department),
        )
        conn.commit()
        conn.close()

        flash("Class created successfully!", "success")
        return redirect(url_for("manage_classes"))

    conn = get_db_connection()
    
    # Ensure department column exists
    try:
        classes = conn.execute(
            "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
    except:
        conn.execute("ALTER TABLE classes ADD COLUMN department TEXT")
        conn.commit()
        classes = conn.execute(
            "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
        ).fetchall()
    
    # Get student counts for each class
    classes_with_counts = []
    for class_row in classes:
        student_count = conn.execute(
            "SELECT COUNT(*) as count FROM class_students WHERE class_id = ?",
            (class_row["id"],)
        ).fetchone()["count"]
        
        class_dict = dict(class_row)
        class_dict["student_count"] = student_count
        classes_with_counts.append(class_dict)
    
    conn.close()

    return render_template("teacher/classes.html", classes=classes_with_counts)


@app.route("/teacher/class/<int:class_id>")
def view_class(class_id):
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    conn = get_db_connection()
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Get students in this class
    students = conn.execute(
        "SELECT student_name, added_at FROM class_students WHERE class_id = ? ORDER BY student_name",
        (class_id,)
    ).fetchall()

    # Get attendance records from attendance_records table
    attendance_records = conn.execute(
        """SELECT a.id, a.student_name, a.class_id, a.teacher_id, a.date, a.time, a.status
           FROM attendance_records a
           WHERE a.class_id = ? AND a.teacher_id = ?
           ORDER BY a.date DESC, a.time DESC LIMIT 50""",
        (class_id, session["teacher_id"]),
    ).fetchall()

    # Get all registered users (for adding students to class)
    all_students = conn.execute(
        "SELECT DISTINCT student_name FROM class_students ORDER BY student_name"
    ).fetchall()

    conn.close()

    # Serialize attendance for JSON
    attendance = [
        {
            "id": record["id"],
            "student_name": record["student_name"],
            "class_id": record["class_id"],
            "teacher_id": record["teacher_id"],
            "date": str(record["date"]),
            "time": str(record["time"]),
            "status": record["status"],
        }
        for record in attendance_records
    ]

    return render_template(
        "teacher/class_detail.html", 
        class_info=class_info, 
        attendance=attendance,
        students=students,
        all_students=all_students
    )


@app.route("/teacher/add_student_to_class/<int:class_id>", methods=["POST"])
def add_student_to_class(class_id):
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    student_name = request.form.get("student_name", "").strip()
    if not student_name:
        return (
            jsonify({"success": False, "message": "Student name cannot be empty"}),
            400,
        )

    conn = get_db_connection()
    try:
        # Verify class ownership
        class_info = conn.execute(
            "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
            (class_id, session["teacher_id"]),
        ).fetchone()

        if not class_info:
            return (
                jsonify(
                    {"success": False, "message": "Class not found or access denied"}
                ),
                404,
            )

        # Check if student exists in face data
        if os.path.exists("data/names.pkl"):
            with open("data/names.pkl", "rb") as f:
                existing_students = pickle.load(f)
            if student_name not in existing_students:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f'Student "{student_name}" needs face registration first',
                        }
                    ),
                    400,
                )

        # Add to class
        conn.execute(
            "INSERT INTO class_students (class_id, student_name) VALUES (?, ?)",
            (class_id, student_name),
        )
        conn.commit()
        return jsonify(
            {
                "success": True,
                "message": f"{student_name} added to class successfully",
                "student": student_name,
            }
        )

    except sqlite3.IntegrityError:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"{student_name} is already in this class",
                }
            ),
            409,
        )
    except Exception as e:
        print(f"Error adding student: {str(e)}")
        return jsonify({"success": False, "message": "Database error occurred"}), 500
    finally:
        conn.close()


@app.route("/teacher/remove_student_from_class/<int:class_id>", methods=["POST"])
def remove_student_from_class(class_id):
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    student_name = request.form.get("student_name", "").strip()
    if not student_name:
        return jsonify({"success": False, "message": "Student name required"}), 400

    conn = get_db_connection()
    try:
        # Verify class ownership
        class_info = conn.execute(
            "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
            (class_id, session["teacher_id"]),
        ).fetchone()

        if not class_info:
            return jsonify({"success": False, "message": "Class not found or access denied"}), 404

        # Remove student from class
        result = conn.execute(
            "DELETE FROM class_students WHERE class_id = ? AND student_name = ?",
            (class_id, student_name),
        )
        conn.commit()

        if result.rowcount > 0:
            return jsonify({
                "success": True,
                "message": f"{student_name} removed from class successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": f"{student_name} not found in this class"
            }), 404

    except Exception as e:
        print(f"Error removing student: {str(e)}")
        return jsonify({"success": False, "message": "Database error occurred"}), 500
    finally:
        conn.close()


@app.route("/teacher/delete_class/<int:class_id>")
def delete_class(class_id):
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    conn = get_db_connection()
    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Delete class and associated records
    conn.execute("DELETE FROM class_students WHERE class_id = ?", (class_id,))
    conn.execute("DELETE FROM attendance_records WHERE class_id = ?", (class_id,))
    conn.execute("DELETE FROM classes WHERE id = ?", (class_id,))
    conn.commit()
    conn.close()

    flash("Class deleted successfully.", "success")
    return redirect(url_for("manage_classes"))


# Enhanced attendance views
@app.route("/teacher/attendance")
def teacher_attendance():
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Get all classes for filter dropdown
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (teacher_id,)
    ).fetchall()

    # Get all students for this teacher
    students_query = """
    SELECT DISTINCT cs.student_name 
    FROM class_students cs
    JOIN classes c ON cs.class_id = c.id
    WHERE c.teacher_id = ?
    """
    students = conn.execute(students_query, (teacher_id,)).fetchall()

    # Default to showing today's attendance
    today = datetime.now().strftime("%d-%m-%Y")

    # Get attendance for today
    attendance_query = """
    SELECT ar.*, c.name as class_name 
    FROM attendance_records ar
    LEFT JOIN classes c ON ar.class_id = c.id
    WHERE ar.teacher_id = ? AND ar.date = ?
    ORDER BY ar.time DESC
    """
    attendance = conn.execute(attendance_query, (teacher_id, today)).fetchall()

    conn.close()

    return render_template(
        "teacher/attendance.html",
        classes=classes,
        students=students,
        attendance=attendance,
        selected_date=today,
    )


@app.route("/teacher/class/<int:class_id>/student_attendance")
def class_student_attendance(class_id):
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?", (class_id, teacher_id)
    ).fetchone()

    if not class_info:
        conn.close()
        flash("Class not found or access denied.", "danger")
        return redirect(url_for("manage_classes"))

    # Get all students in this class
    students = conn.execute(
        "SELECT * FROM class_students WHERE class_id = ?", (class_id,)
    ).fetchall()

    # Get student names
    student_names = [student["student_name"] for student in students]

    # Default to showing current month's attendance
    today = datetime.now()
    start_of_month = today.replace(day=1).strftime("%d-%m-%Y")
    end_of_month = today.strftime("%d-%m-%Y")

    # Get attendance records for all students in this class
    attendance_records = {}
    attendance_dates = set()

    for student_name in student_names:
        # Get attendance records for this student
        query = """
        SELECT * FROM attendance_records 
        WHERE class_id = ? AND teacher_id = ? AND student_name = ?
        ORDER BY date ASC, time ASC
        """
        student_records = conn.execute(
            query, (class_id, teacher_id, student_name)
        ).fetchall()

        # Convert to dict for easier access
        attendance_records[student_name] = {}
        for record in student_records:
            date = record["date"]
            attendance_records[student_name][date] = record
            attendance_dates.add(date)

    # Sort dates
    sorted_dates = sorted(list(attendance_dates))

    # Generate attendance summary
    attendance_summary = []
    for student_name in student_names:
        present_count = sum(
            1 for date in sorted_dates if date in attendance_records[student_name]
        )
        attendance_rate = present_count / len(sorted_dates) * 100 if sorted_dates else 0

        student_summary = {
            "name": student_name,
            "present_count": present_count,
            "total_days": len(sorted_dates),
            "attendance_rate": round(attendance_rate, 1),
        }
        attendance_summary.append(student_summary)

    conn.close()

    return render_template(
        "teacher/student_attendance.html",
        class_info=class_info,
        students=students,
        attendance_records=attendance_records,
        dates=sorted_dates,
        attendance_summary=attendance_summary,
    )


@app.route("/teacher/class/<int:class_id>/student_attendance/filter", methods=["POST"])
def filter_class_student_attendance(class_id):
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    student_name = request.form.get("student_name")

    conn = get_db_connection()

    # Verify class belongs to teacher
    class_info = conn.execute(
        "SELECT * FROM classes WHERE id = ? AND teacher_id = ?",
        (class_id, session["teacher_id"]),
    ).fetchone()

    if not class_info:
        conn.close()
        return (
            jsonify({"success": False, "message": "Class not found or access denied"}),
            403,
        )

    # Base query for student attendance
    query = """
    SELECT * FROM attendance_records 
    WHERE class_id = ? AND teacher_id = ?
    """
    params = [class_id, session["teacher_id"]]

    # Add date range filter if provided
    if start_date and end_date:
        query += " AND date BETWEEN ? AND ?"
        params.extend([start_date, end_date])

    # Add student filter if provided
    if student_name:
        query += " AND student_name = ?"
        params.append(student_name)

    query += " ORDER BY date ASC, time ASC"

    filtered_records = conn.execute(query, params).fetchall()

    # Get unique dates and students
    dates = set()
    students = set()

    for record in filtered_records:
        dates.add(record["date"])
        students.add(record["student_name"])

    # Sort dates
    sorted_dates = sorted(list(dates))
    sorted_students = sorted(list(students))

    # Organize by student and date
    attendance_records = {}
    for student_name in sorted_students:
        attendance_records[student_name] = {}

    for record in filtered_records:
        student_name = record["student_name"]
        date = record["date"]
        attendance_records[student_name][date] = record

    # Generate attendance summary
    attendance_summary = []
    for student_name in sorted_students:
        present_count = sum(
            1 for date in sorted_dates if date in attendance_records[student_name]
        )
        attendance_rate = present_count / len(sorted_dates) * 100 if sorted_dates else 0

        student_summary = {
            "name": student_name,
            "present_count": present_count,
            "total_days": len(sorted_dates),
            "attendance_rate": round(attendance_rate, 1),
        }
        attendance_summary.append(student_summary)

    # Convert to JSON-serializable format
    result = {
        "success": True,
        "dates": sorted_dates,
        "students": sorted_students,
        "attendance_records": {
            student: {
                date: dict(attendance_records[student][date])
                for date in attendance_records[student]
            }
            for student in attendance_records
        },
        "attendance_summary": attendance_summary,
    }

    conn.close()
    return jsonify(result)


@app.route("/teacher/attendance/filter", methods=["POST"])
def filter_attendance():
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    # Handle both form data and JSON
    if request.is_json:
        data = request.get_json()
        date_filter = data.get("date")
        class_filter = data.get("class_id")
        student_filter = data.get("student_name")
    else:
        date_filter = request.form.get("date")
        class_filter = request.form.get("class_id")
        student_filter = request.form.get("student_name")

    query = "SELECT ar.*, c.name as class_name FROM attendance_records ar LEFT JOIN classes c ON ar.class_id = c.id WHERE ar.teacher_id = ?"
    params = [session["teacher_id"]]

    if date_filter:
        query += " AND ar.date = ?"
        params.append(date_filter)

    if class_filter:
        query += " AND ar.class_id = ?"
        params.append(int(class_filter))

    if student_filter:
        query += " AND ar.student_name = ?"
        params.append(student_filter)

    query += " ORDER BY ar.date DESC, ar.time DESC"

    conn = get_db_connection()
    attendance = conn.execute(query, params).fetchall()
    conn.close()

    # Convert to list of dicts for JSON response
    result = [dict(row) for row in attendance]
    return jsonify({"success": True, "attendance": result})


@app.route("/teacher/export_today")
def export_today_attendance():
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    # Get today's date
    today = datetime.now().strftime("%d-%m-%Y")

    # Query for today's attendance records
    query = """
    SELECT ar.*, c.name as class_name 
    FROM attendance_records ar 
    LEFT JOIN classes c ON ar.class_id = c.id 
    WHERE ar.teacher_id = ? AND ar.date = ? 
    ORDER BY ar.class_id, ar.student_name, ar.time
    """

    conn = get_db_connection()
    attendance = conn.execute(query, (session["teacher_id"], today)).fetchall()
    conn.close()

    if not attendance:
        flash("No attendance records for today.", "warning")
        return redirect(url_for("teacher_dashboard"))

    # Create CSV file
    filename = f"today_attendance_{today}.csv"
    filepath = os.path.join("data", filename)

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Student Name", "Class", "Time", "Status"])

        for record in attendance:
            writer.writerow(
                [
                    record["student_name"],
                    record["class_name"] or "No Class",
                    record["time"],
                    record["status"],
                ]
            )

    flash("Today's attendance exported successfully!", "success")
    return send_from_directory("data", filename, as_attachment=True)


@app.route("/teacher/export", methods=["POST"])
def teacher_export_attendance():
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    # Handle both form data and JSON
    if request.is_json:
        data = request.get_json()
        date_filter = data.get("date")
        class_filter = data.get("class_id")
        student_filter = data.get("student_name")
    else:
        date_filter = request.form.get("date")
        class_filter = request.form.get("class_id")
        student_filter = request.form.get("student_name")

    query = "SELECT ar.*, c.name as class_name FROM attendance_records ar LEFT JOIN classes c ON ar.class_id = c.id WHERE ar.teacher_id = ?"
    params = [session["teacher_id"]]

    if date_filter:
        query += " AND ar.date = ?"
        params.append(date_filter)

    if class_filter:
        query += " AND ar.class_id = ?"
        params.append(int(class_filter))

    if student_filter:
        query += " AND ar.student_name = ?"
        params.append(student_filter)

    query += " ORDER BY ar.date DESC, ar.time DESC"

    conn = get_db_connection()
    attendance = conn.execute(query, params).fetchall()
    conn.close()

    if not attendance:
        return jsonify({"success": False, "message": "No attendance data to export"}), 404

    # Create CSV file in memory
    from io import StringIO
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student Name", "Class", "Date", "Time", "Status", "Notes"])

    for record in attendance:
        writer.writerow(
            [
                record["student_name"],
                record["class_name"] or "No Class",
                record["date"],
                record["time"],
                record["status"],
                record.get("notes", "") or "",
            ]
        )

    # Create response
    output.seek(0)
    filename = f'attendance_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv'
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# Delete attendance record endpoint
@app.route("/teacher/delete_record/<int:record_id>", methods=["DELETE", "POST"])
def delete_attendance_record(record_id):
    if "teacher_id" not in session:
        return jsonify({"success": False, "message": "Authentication required"}), 401

    teacher_id = session["teacher_id"]
    conn = get_db_connection()

    # Verify the record belongs to this teacher
    record = conn.execute(
        "SELECT * FROM attendance_records WHERE id = ? AND teacher_id = ?",
        (record_id, teacher_id)
    ).fetchone()

    if not record:
        conn.close()
        return jsonify({"success": False, "message": "Record not found or access denied"}), 404

    # Delete the record
    conn.execute("DELETE FROM attendance_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": "Record deleted successfully"})


# Store attendance in database when recognition happens
def store_attendance_in_db(student_name, date, time_str, class_id=None):
    print(f"\n*** STORING ATTENDANCE: {student_name}, {date}, {time_str} ***")
    try:
        # Set default values
        status = "Present"
        teacher_id = None

        # If a teacher is logged in, use their ID
        if "teacher_id" in session:
            teacher_id = session["teacher_id"]
            print(f"Using teacher ID {teacher_id} from session")

            # Look for class_id if not provided and teacher is logged in
            if class_id is None and student_name:
                conn = get_db_connection()
                # Try to find a class this student belongs to
                class_result = conn.execute(
                    """SELECT class_id FROM class_students 
                       WHERE student_name = ? AND class_id IN 
                       (SELECT id FROM classes WHERE teacher_id = ?)""",
                    (student_name, teacher_id),
                ).fetchone()

                if class_result:
                    class_id = class_result["class_id"]
                    print(f"Found class {class_id} for student {student_name}")
                conn.close()
        else:
            # If no teacher is logged in, use a default teacher ID of 1
            teacher_id = 1
            print(
                f"No teacher logged in, using default teacher ID 1 for {student_name}"
            )

        # Make sure we have a valid teacher_id
        if not teacher_id:
            teacher_id = 1
            print(f"Teacher ID was None, using default value 1")

        # Insert attendance record
        conn = get_db_connection()

        # Check if there are any existing teachers in the database
        teacher_check = conn.execute(
            "SELECT COUNT(*) as count FROM teachers"
        ).fetchone()
        print(f"Found {teacher_check['count']} teachers in database")

        # If no teacher exists and we're using the default ID, create a default teacher
        if teacher_check["count"] == 0 and teacher_id == 1:
            try:
                conn.execute(
                    "INSERT INTO teachers (id, username, password_hash, email, full_name) VALUES (?, ?, ?, ?, ?)",
                    (
                        1,
                        "default",
                        "default_hash",
                        "default@example.com",
                        "Default Teacher",
                    ),
                )
                conn.commit()
                print("Created default teacher for attendance records")
            except sqlite3.IntegrityError as e:
                print(
                    f"Error creating default teacher: {str(e)}, but continuing anyway"
                )

        # Debug the SQL command we're about to execute
        print(
            f"Executing INSERT with values: ({student_name}, {class_id}, {teacher_id}, {date}, {time_str}, {status})"
        )

        # Insert the attendance record
        conn.execute(
            "INSERT INTO attendance_records (student_name, class_id, teacher_id, date, time, status) VALUES (?, ?, ?, ?, ?, ?)",
            (student_name, class_id, teacher_id, date, time_str, status),
        )
        conn.commit()

        # Verify the record was inserted
        verify = conn.execute(
            "SELECT id FROM attendance_records WHERE student_name = ? AND date = ? AND time = ?",
            (student_name, date, time_str),
        ).fetchone()

        conn.close()

        if verify:
            print(
                f"✓ Successfully stored attendance record #{verify['id']} for {student_name} in database"
            )
            return True
        else:
            print(f"✗ Failed to verify attendance record was stored for {student_name}")
            return False

    except Exception as e:
        print(f"ERROR STORING ATTENDANCE: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# Note: We're using the original recognize route defined above
# This comment replaces the duplicate route


# Test route to manually create an attendance record for debugging
@app.route("/test_attendance")
def test_attendance():
    student_name = "Test Student"
    date = datetime.now().strftime("%d-%m-%Y")
    time_str = datetime.now().strftime("%H:%M:%S")

    print(f"\n********* CREATING TEST ATTENDANCE RECORD *********")
    success = store_attendance_in_db(student_name, date, time_str, None)

    # Also check database records
    conn = get_db_connection()
    records = conn.execute("SELECT * FROM attendance_records").fetchall()
    conn.close()

    info = f"Test record created: {success}\n"
    info += f"Total records in database: {len(records)}\n"

    for record in records:
        info += f"Record #{record['id']}: {record['student_name']} on {record['date']} at {record['time']}\n"

    return render_template("result.html", result=info)


if __name__ == "__main__":
    app.run(debug=True)
