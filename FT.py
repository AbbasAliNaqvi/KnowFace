import cv2
import face_recognition
import numpy as np
import os
import time
import pyttsx3
from pathlib import Path
import threading
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, database_path, confidence_threshold=0.6):
        """
        Initialize the face recognition system.
        
        Args:
            database_path (str): Path to the folder containing face images
            confidence_threshold (float): Threshold for face match confidence (0-1)
        """
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # UI customization properties
        self.ui_colors = {
            'background': (40, 44, 52),  # Dark blue-gray
            'text': (236, 240, 241),     # Light gray
            'recognized': (46, 204, 113), # Green
            'processing': (52, 152, 219), # Blue
            'unknown': (231, 76, 60),    # Red
            'header': (142, 68, 173)     # Purple
        }
        
        # Tracking variables for recognized faces
        self.recognized_faces = {}  # Format: {name: {"last_seen": timestamp, "count": int}}
        
        # Load the face database
        self.load_face_database()
        
    def load_face_database(self):
        """Load all face images from the database directory."""
        print(f"Loading face database from: {self.database_path}")
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        # Iterate through all files in the database directory
        for file in os.listdir(self.database_path):
            path = os.path.join(self.database_path, file)
            
            # Check if it's a file with supported extension
            if os.path.isfile(path) and Path(file).suffix.lower() in image_extensions:
                # Extract name from filename (without extension)
                name = os.path.splitext(file)[0]
                
                try:
                    # Load image and get face encoding
                    image = face_recognition.load_image_file(path)
                    face_locations = face_recognition.face_locations(image, model="hog")  # Use HOG for faster processing
                    
                    if face_locations:
                        # Get the encoding of the first face found
                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                        
                        # Add to our lists
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        
                        print(f"Added {name} to the database")
                    else:
                        print(f"No face found in {file}, skipping...")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        print(f"Database loaded with {len(self.known_face_names)} faces")
        
        # Announce database loading complete
        self.speak(f"Database loaded with {len(self.known_face_names)} faces")
    
    def speak(self, text):
        """Use text-to-speech to speak the given text."""
        try:
            # Directly speak without threading to avoid runAndWait issues
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError as e:
            # If run loop already started, just print the message
            print(f"Speech: {text}")
    
    def create_ui_frame(self, frame, fps=0):
        """Create a styled UI frame with info panel."""
        # Create a blank frame with our background color
        h, w = frame.shape[:2]
        info_panel_height = 80
        ui_frame = np.zeros((h + info_panel_height, w, 3), dtype=np.uint8)
        
        # Fill with background color
        ui_frame[:] = self.ui_colors['background']
        
        # Add the video frame
        ui_frame[0:h, 0:w] = frame
        
        # Add a header bar (use fillPoly instead of rectangle)
        pts = np.array([[0, h], [w, h], [w, h + info_panel_height], [0, h + info_panel_height]])
        cv2.fillPoly(ui_frame, [pts], self.ui_colors['header'])
        
        # Add system info and stats
        current_time = datetime.now().strftime("%H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Add current time and date
        cv2.putText(ui_frame, f"Time: {current_time}", (10, h + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        cv2.putText(ui_frame, f"Date: {date_str}", (200, h + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Add FPS counter
        cv2.putText(ui_frame, f"FPS: {fps:.1f}", (400, h + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Add database info
        cv2.putText(ui_frame, f"Database: {len(self.known_face_names)} faces", (10, h + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Add controls info
        cv2.putText(ui_frame, "Press 'q' to quit | 's' to screenshot", (400, h + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        return ui_frame
    
    def draw_face_box(self, frame, face_location, name, confidence=None):
        """Draw a styled box around a face with name and confidence."""
        top, right, bottom, left = face_location
        
        # Determine box color based on recognition status
        if name == "Unknown":
            color = self.ui_colors['unknown']
        elif name == "Processing...":
            color = self.ui_colors['processing']
        else:
            color = self.ui_colors['recognized']
        
        # Draw a rectangle around the face - ensure integer coordinates
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Create a nicer label area
        label_y = bottom + 30
        if label_y >= frame.shape[0]:
            label_y = bottom - 10  # Show above if at bottom edge
            
        # Draw a filled rectangle for the label background
        label_text = name
        if confidence is not None:
            label_text = f"{name} ({confidence:.1f}%)"
        
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (left, bottom), (left + label_size[0] + 10, bottom + 30), color, cv2.FILLED)
        
        # Draw the label text
        cv2.putText(frame, label_text, (left + 5, bottom + 25), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Run the face recognition system with webcam input."""
        if not self.known_face_encodings:
            print("No faces in database. Please add images to the database folder.")
            self.speak("No faces in database. Please add images to the database folder.")
            return
        
        # Initialize webcam
        print("Starting webcam...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            self.speak("Could not open webcam.")
            return
        
        # Set webcam properties for better quality (if supported)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Let the camera warm up
        time.sleep(1)
        
        print("Face recognition system running. Press 'q' to quit, 's' to save screenshot.")
        self.speak("Face recognition system is now running")
        
        # Variables to control recognition frequency
        last_recognition_time = 0
        recognition_cooldown = 0.5  # seconds between recognition attempts
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Create window with custom name and properties
        window_name = "Advanced Face Recognition System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break
            
            # Only run face recognition every few frames to improve performance
            current_time = time.time()
            process_this_frame = (current_time - last_recognition_time) > recognition_cooldown
            
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            # Track currently detected faces
            current_faces = {}
            
            if process_this_frame:
                # Convert the image from BGR color (which OpenCV uses) to RGB color
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find all the faces in the current frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # Get encodings for detected faces
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Process each detected face
                    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                        # Calculate distances to all known faces
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        
                        name = "Unknown"
                        confidence = 0
                        
                        if len(face_distances) > 0:
                            # Find the best match (lowest distance)
                            best_index = np.argmin(face_distances)
                            best_distance = face_distances[best_index]
                            
                            # Check if it's a good enough match
                            if best_distance < self.confidence_threshold:
                                name = self.known_face_names[best_index]
                                confidence = (1 - best_distance) * 100
                        
                        # Store the recognized face
                        current_faces[i] = {
                            "name": name,
                            "location": face_location,
                            "confidence": confidence
                        }
                
                # Update last recognition time
                last_recognition_time = current_time
            
            # Process announcements and draw face boxes
            for face_id, face_data in current_faces.items():
                name = face_data["name"]
                location = face_data["location"]
                confidence = face_data["confidence"]
                
                # Draw the face box on the display frame
                display_frame = self.draw_face_box(display_frame, location, name, confidence)
                
                # Handle recognized face announcements
                if name != "Unknown" and confidence > 0:
                    # Check if this is a new recognition or after cooldown
                    current_time = time.time()
                    if name not in self.recognized_faces:
                        self.recognized_faces[name] = {
                            "last_seen": current_time,
                            "count": 1
                        }
                        # Announce the first recognition
                        self.speak(f"Recognized {name}")
                    else:
                        # Update the last seen time
                        self.recognized_faces[name]["last_seen"] = current_time
                        self.recognized_faces[name]["count"] += 1
                        
                        # Announce again if it's been a while (30 seconds)
                        if current_time - self.recognized_faces[name]["last_seen"] > 30:
                            self.speak(f"Recognized {name} again")
            
            # Create the UI frame with system info
            ui_frame = self.create_ui_frame(display_frame, fps)
            
            # Show the recognition status
            recognized_count = sum(1 for face in current_faces.values() if face["name"] != "Unknown")
            total_faces = len(current_faces)
            
            status_text = f"Recognized: {recognized_count}/{total_faces} faces"
            cv2.putText(ui_frame, status_text, (ui_frame.shape[1] - 300, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors['text'], 2)
            
            # Display the resulting frame
            cv2.imshow(window_name, ui_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save a screenshot
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, ui_frame)
                print(f"Screenshot saved as {filename}")
                self.speak(f"Screenshot saved")
        
        # Release the webcam and close all windows
        video_capture.release()
        cv2.destroyAllWindows()
        print("Face recognition system stopped.")
        self.speak("Face recognition system stopped")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Advanced Face Recognition System')
    parser.add_argument('--database', '-d', default='faces_db', 
                        help='Path to the folder containing face images (default: faces_db)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Face matching confidence threshold (default: 0.6, lower is stricter)')
    
    args = parser.parse_args()
    
    # Create and run the face recognition system
    face_system = FaceRecognitionSystem(args.database, args.threshold)
    face_system.run()