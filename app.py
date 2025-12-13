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
    # Initialize empty list for attendance records
    attendance_records = []

    # Add header row
    attendance_records.append(["NAME", "DATE", "TIME"])

    # Check the database first (more reliable source)
    try:
        conn = get_db_connection()
        # Query all attendance records from the database, sorted by most recent first
        db_records = conn.execute(
            """
            SELECT student_name, date, time 
            FROM attendance_records 
            WHERE student_name != "Unknown" AND student_name != "Error"
            ORDER BY date DESC, time DESC
        """
        ).fetchall()
        conn.close()

        if db_records:
            # Convert database records to the right format
            for record in db_records:
                attendance_records.append(
                    [record["student_name"], record["date"], record["time"]]
                )
            print(f"Found {len(db_records)} records in database")
    except Exception as e:
        print(f"Error querying database for attendance records: {str(e)}")

    # Also check CSV files for any records that might not be in the database
    date = datetime.now().strftime("%d-%m-%Y")
    file_path = os.path.join(
        r"C:\Users\MANISH SHARMA\OneDrive\Desktop\Smart Attendence System\Attendance",
        f"Attendance_{date}.csv",
    )
    if os.path.isfile(file_path):
        with open(file_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            csv_records = list(reader)

            # Skip header row if it exists
            start_idx = 1 if len(csv_records) > 0 and csv_records[0][0] == "NAME" else 0

            # Add CSV records (avoiding duplicates)
            db_record_strings = (
                [f"{r[0]}-{r[1]}-{r[2]}" for r in attendance_records[1:]]
                if len(attendance_records) > 1
                else []
            )

            for i in range(start_idx, len(csv_records)):
                record = csv_records[i]
                if len(record) >= 3:
                    record_string = f"{record[0]}-{record[1]}-{record[2]}"
                    if record_string not in db_record_strings:
                        attendance_records.append(record)
            print(f"Added {len(csv_records) - start_idx} unique records from CSV file")

    # If we only have the header row, there are no records
    has_records = len(attendance_records) > 1

    return render_template(
        "attendance.html", attendance_data=attendance_records, has_records=has_records
    )


@app.route("/manage_users")
def manage_users():
    users = get_all_users()
    return render_template("manage_users.html", users=users)


@app.route("/delete_user/<username>")
def delete_user_route(username):
    if delete_user(username):
        return redirect(url_for("manage_users"))
    return "Error deleting user", 400


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
        return redirect(url_for("settings"))

    current_settings = load_settings()
    return render_template("settings.html", settings=current_settings)


@app.route("/export_attendance")
def export_attendance():
    filename = export_attendance_csv()
    if filename:
        return send_from_directory("data", filename, as_attachment=True)
    return "No attendance data to export", 400


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

    # Get recent attendance records
    recent_attendance = conn.execute(
        """SELECT ar.*, c.name as class_name 
           FROM attendance_records ar 
           LEFT JOIN classes c ON ar.class_id = c.id 
           WHERE ar.teacher_id = ? 
           ORDER BY ar.date DESC, ar.time DESC LIMIT 10""",
        (teacher_id,),
    ).fetchall()

    conn.close()

    return render_template(
        "teacher/dashboard.html", classes=classes, recent_attendance=recent_attendance
    )


# Class management routes
@app.route("/teacher/classes", methods=["GET", "POST"])
def manage_classes():
    if "teacher_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

    if request.method == "POST":
        name = request.form["name"]
        description = request.form["description"]
        teacher_id = session["teacher_id"]

        conn = get_db_connection()
        conn.execute(
            "INSERT INTO classes (teacher_id, name, description) VALUES (?, ?, ?)",
            (teacher_id, name, description),
        )
        conn.commit()
        conn.close()

        flash("Class created successfully!", "success")
        return redirect(url_for("manage_classes"))

    conn = get_db_connection()
    classes = conn.execute(
        "SELECT * FROM classes WHERE teacher_id = ?", (session["teacher_id"],)
    ).fetchall()
    conn.close()

    return render_template("teacher/classes.html", classes=classes)


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

    # Get attendance records from attendance_records table
    attendance_records = conn.execute(
        """SELECT a.id, a.student_name, a.class_id, a.teacher_id, a.date, a.time, a.status
           FROM attendance_records a
           WHERE a.class_id = ? AND a.teacher_id = ?
           ORDER BY a.date DESC, a.time DESC""",
        (class_id, session["teacher_id"]),
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
        "teacher/class_detail.html", class_info=class_info, attendance=attendance
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
        params.append(class_filter)

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
        flash("Please log in first.", "warning")
        return redirect(url_for("teacher_login"))

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
        params.append(class_filter)

    if student_filter:
        query += " AND ar.student_name = ?"
        params.append(student_filter)

    query += " ORDER BY ar.date DESC, ar.time DESC"

    conn = get_db_connection()
    attendance = conn.execute(query, params).fetchall()
    conn.close()

    if not attendance:
        flash("No attendance data to export.", "warning")
        return redirect(url_for("teacher_attendance"))

    # Create CSV file
    filename = f'teacher_attendance_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.csv'
    filepath = os.path.join("data", filename)

    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Student Name", "Class", "Date", "Time", "Status", "Notes"])

        for record in attendance:
            writer.writerow(
                [
                    record["student_name"],
                    record["class_name"] or "No Class",
                    record["date"],
                    record["time"],
                    record["status"],
                    record["notes"] or "",
                ]
            )

    return send_from_directory("data", filename, as_attachment=True)


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
