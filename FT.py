import cv2
import face_recognition
import numpy as np
import os
import time
import pyttsx3
from pathlib import Path
import threading
from datetime import datetime
import random

class SecureAuthentication:
    def __init__(self, database_path, confidence_threshold=0.6):
        """
        Initialize the secure authentication system with cinematic UI.
        
        Args:
            database_path (str): Path to the folder containing face images
            confidence_threshold (float): Threshold for face match confidence (0-1)
        """
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.employee_images = {}  # Store face images for database display
        self.employee_data = {}    # Store employee data (ID, clearance, etc)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # UI customization properties
        self.ui_colors = {
            'background': (10, 10, 20),         # Almost black with blue tint
            'text': (0, 255, 255),              # Cyan
            'recognized': (0, 255, 0),          # Green
            'processing': (0, 140, 255),        # Orange
            'unknown': (0, 0, 255),             # Red (BGR)
            'header': (139, 0, 139),            # Dark magenta
            'scan_line': (0, 255, 255),         # Cyan
            'grid': (20, 80, 20),               # Dark green grid
            'access_granted': (0, 255, 0),      # Green
            'access_denied': (0, 0, 255)        # Red
        }
        
        # Authentication states
        self.state = "LOCKED"  # States: LOCKED, SCANNING, ACCESS_GRANTED, ACCESS_DENIED, DATABASE
        self.scan_progress = 0
        self.scan_line_pos = 0
        self.auth_message = ""
        
        # Load the face database
        self.load_face_database()
        
    def load_face_database(self):
        """Load all face images from the database directory."""
        print(f"Loading secure database from: {self.database_path}")
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            print(f"Created database directory: {self.database_path}")
            
        # Iterate through all files in the database directory
        for file in os.listdir(self.database_path):
            path = os.path.join(self.database_path, file)
            
            # Check if it's a file with supported extension
            if os.path.isfile(path) and Path(file).suffix.lower() in image_extensions:
                # Extract name from filename (without extension)
                name = os.path.splitext(file)[0]
                
                try:
                    # Load image for database display and face recognition
                    image = face_recognition.load_image_file(path)
                    image_for_display = cv2.imread(path)
                    
                    face_locations = face_recognition.face_locations(image)
                    
                    if face_locations:
                        # Get the encoding of the first face found
                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                        
                        # Add to our lists
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        
                        # Store the image for database display
                        self.employee_images[name] = image_for_display
                        
                        # Create fake employee data
                        self.employee_data[name] = {
                            "id": f"EMP-{random.randint(1000, 9999)}",
                            "clearance": random.choice(["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5"]),
                            "division": random.choice(["RESEARCH", "SECURITY", "OPERATIONS", "COMMAND", "INTELLIGENCE"]),
                            "status": random.choice(["ACTIVE", "ON MISSION", "IN FACILITY"]),
                        }
                        
                        print(f"Added {name} to the secure database")
                    else:
                        print(f"No face found in {file}, skipping...")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        print(f"Secure database loaded with {len(self.known_face_names)} personnel")
        
        # Create sample data if database is empty
        if len(self.known_face_names) == 0:
            print("Database empty. Please add face images to the database folder.")
    
    def speak(self, text):
        """Use text-to-speech to speak the given text."""
        try:
            # Use a safer approach that avoids threading issues
            print(f"Speech: {text}")
            
            # Only use speech if not already speaking
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError:
                # If run loop already started, just print the message
                pass
            except Exception as e:
                print(f"Speech engine error: {e}")
        except Exception as e:
            print(f"Speech function error: {e}")
    
    def create_locked_screen(self, frame_width, frame_height):
        """Create the locked device screen with cinematic effects."""
        # Create a black background
        screen = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Add grid pattern
        grid_spacing = 30
        grid_color = self.ui_colors['grid']
        
        # Draw vertical grid lines
        for x in range(0, frame_width, grid_spacing):
            cv2.line(screen, (x, 0), (x, frame_height), grid_color, 1)
            
        # Draw horizontal grid lines
        for y in range(0, frame_height, grid_spacing):
            cv2.line(screen, (0, y), (frame_width, y), grid_color, 1)
        
        # Add logo and header text
        logo_text = "SECURE AUTHENTICATION SYSTEM"
        cv2.putText(screen, logo_text, (int(frame_width/2) - 250, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.ui_colors['text'], 2)
        
        # Add facility name
        facility_text = "RESTRICTED ACCESS FACILITY"
        cv2.putText(screen, facility_text, (int(frame_width/2) - 200, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors['text'], 1)
        
        # Add authentication prompt
        auth_text = "BIOMETRIC AUTHENTICATION REQUIRED"
        cv2.putText(screen, auth_text, (int(frame_width/2) - 220, frame_height - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors['text'], 1)
        
        # Display current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(screen, f"SYSTEM TIME: {current_time}", (20, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Add scanning button instruction
        instruction_text = "PRESS 'SPACE' TO INITIATE FACIAL SCAN"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = int((frame_width - text_size[0]) / 2)
        
        # Create flashing effect
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(screen, instruction_text, (text_x, frame_height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_colors['text'], 2)
        
        # Add security level indicator
        security_text = "SECURITY LEVEL: MAXIMUM"
        cv2.putText(screen, security_text, (frame_width - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_colors['text'], 1)
        
        # Add decorative elements (corners)
        corner_size = 40
        line_thickness = 2
        
        # Top-left corner
        cv2.line(screen, (0, 0), (corner_size, 0), self.ui_colors['text'], line_thickness)
        cv2.line(screen, (0, 0), (0, corner_size), self.ui_colors['text'], line_thickness)
        
        # Top-right corner
        cv2.line(screen, (frame_width-1, 0), (frame_width-corner_size-1, 0), self.ui_colors['text'], line_thickness)
        cv2.line(screen, (frame_width-1, 0), (frame_width-1, corner_size), self.ui_colors['text'], line_thickness)
        
        # Bottom-left corner
        cv2.line(screen, (0, frame_height-1), (corner_size, frame_height-1), self.ui_colors['text'], line_thickness)
        cv2.line(screen, (0, frame_height-1), (0, frame_height-corner_size-1), self.ui_colors['text'], line_thickness)
        
        # Bottom-right corner
        cv2.line(screen, (frame_width-1, frame_height-1), (frame_width-corner_size-1, frame_height-1), self.ui_colors['text'], line_thickness)
        cv2.line(screen, (frame_width-1, frame_height-1), (frame_width-1, frame_height-corner_size-1), self.ui_colors['text'], line_thickness)
        
        return screen
    
    def create_scanning_screen(self, frame):
        """Create the scanning effect overlay on the camera frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Add scanning line
        scan_y = int(self.scan_line_pos * h)
        cv2.line(overlay, (0, scan_y), (w, scan_y), self.ui_colors['scan_line'], 2)
        
        # Add scanning effect (semi-transparent overlay moving down)
        overlay_top = np.zeros((scan_y, w, 3), dtype=np.uint8)
        overlay_top[:] = self.ui_colors['background']
        
        # Add scan effect with some transparency
        alpha = 0.3
        frame[:scan_y, :] = cv2.addWeighted(frame[:scan_y, :], alpha, overlay_top, 1-alpha, 0)
        
        # Add face tracking rectangles for more cinematic effect
        face_locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for (top, right, bottom, left) in face_locations:
            # Draw main rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), self.ui_colors['processing'], 2)
            
            # Draw corner markers
            corner_length = 20
            # Top-left
            cv2.line(frame, (left, top), (left + corner_length, top), self.ui_colors['text'], 2)
            cv2.line(frame, (left, top), (left, top + corner_length), self.ui_colors['text'], 2)
            # Top-right
            cv2.line(frame, (right, top), (right - corner_length, top), self.ui_colors['text'], 2)
            cv2.line(frame, (right, top), (right, top + corner_length), self.ui_colors['text'], 2)
            # Bottom-left
            cv2.line(frame, (left, bottom), (left + corner_length, bottom), self.ui_colors['text'], 2)
            cv2.line(frame, (left, bottom), (left, bottom - corner_length), self.ui_colors['text'], 2)
            # Bottom-right
            cv2.line(frame, (right, bottom), (right - corner_length, bottom), self.ui_colors['text'], 2)
            cv2.line(frame, (right, bottom), (right, bottom - corner_length), self.ui_colors['text'], 2)
            
            # Add face measurements and data points (for cinematic effect)
            face_width = right - left
            cv2.putText(frame, f"WIDTH: {face_width}px", (right + 10, top + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui_colors['text'], 1)
            
            face_height = bottom - top
            cv2.putText(frame, f"HEIGHT: {face_height}px", (right + 10, top + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui_colors['text'], 1)
            
            # Add random facial feature points
            for _ in range(8):
                x = random.randint(left, right)
                y = random.randint(top, bottom)
                cv2.circle(frame, (x, y), 2, self.ui_colors['text'], -1)
                cv2.circle(frame, (x, y), 4, self.ui_colors['text'], 1)
        
        # Draw progress bar
        progress_width = int(w * 0.6)
        progress_height = 20
        progress_x = int((w - progress_width) / 2)
        progress_y = h - 50
        
        # Draw border
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + progress_width, progress_y + progress_height), 
                     self.ui_colors['text'], 1)
        
        # Draw fill based on scan progress
        fill_width = int(progress_width * self.scan_progress)
        cv2.rectangle(frame, (progress_x, progress_y), 
                     (progress_x + fill_width, progress_y + progress_height), 
                     self.ui_colors['text'], -1)
        
        # Add progress text
        progress_text = f"FACIAL SCAN: {int(self.scan_progress * 100)}%"
        cv2.putText(frame, progress_text, (progress_x, progress_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Add scanning header
        cv2.putText(frame, "FACIAL RECOGNITION IN PROGRESS", (int(w/2) - 220, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors['text'], 2)
        
        # Add system info at the bottom
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"SYSTEM TIME: {current_time}", (20, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['text'], 1)
        
        return frame
    
    def create_access_screen(self, frame, access_granted, identified_name=None):
        """Create the access granted/denied screen."""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = np.zeros_like(frame)
        
        if access_granted:
            # Green overlay for access granted
            overlay[:] = self.ui_colors['access_granted']
            message = "ACCESS GRANTED"
            color = self.ui_colors['access_granted']
            self.auth_message = f"WELCOME, {identified_name}"
        else:
            # Red overlay for access denied
            overlay[:] = self.ui_colors['access_denied']
            message = "ACCESS DENIED"
            color = self.ui_colors['access_denied']
            self.auth_message = "UNAUTHORIZED PERSONNEL"
        
        # Apply overlay with transparency
        alpha = 0.3
        result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        
        # Add the main message
        message_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)[0]
        message_x = int((w - message_size[0]) / 2)
        cv2.putText(result, message, (message_x, int(h/2)), 
                    cv2.FONT_HERSHEY_TRIPLEX, 2, color, 3)
        
        # Add the secondary message (user identification)
        secondary_size = cv2.getTextSize(self.auth_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        secondary_x = int((w - secondary_size[0]) / 2)
        cv2.putText(result, self.auth_message, (secondary_x, int(h/2) + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add instruction
        if access_granted:
            instruction = "PRESS 'D' TO VIEW PERSONNEL DATABASE"
        else:
            instruction = "PRESS 'R' TO RETRY"
            
        instruction_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        instruction_x = int((w - instruction_size[0]) / 2)
        
        # Create flashing effect
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(result, instruction, (instruction_x, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_colors['text'], 1)
        
        return result
    
    def create_database_screen(self, frame_width, frame_height):
        """Create the employee database screen."""
        # Create a dark background
        screen = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        screen[:] = self.ui_colors['background']
        
        # Add header
        cv2.putText(screen, "SECURE PERSONNEL DATABASE", (int(frame_width/2) - 220, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.ui_colors['text'], 2)
        
        # Add system info and controls
        current_time = datetime.now().strftime("%H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        cv2.putText(screen, f"SYSTEM TIME: {current_time}", (20, frame_height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        cv2.putText(screen, f"DATE: {date_str}", (20, frame_height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        cv2.putText(screen, "PRESS 'L' TO LOCK SYSTEM | 'Q' TO QUIT", (frame_width - 400, frame_height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        # Display employee grid
        if len(self.employee_images) == 0:
            # No employees in database
            message = "NO PERSONNEL RECORDS FOUND"
            cv2.putText(screen, message, (int(frame_width/2) - 200, int(frame_height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.ui_colors['text'], 2)
            return screen
        
        # Set up grid layout
        margin = 20
        thumb_width = 150
        thumb_height = 150
        info_height = 80
        cell_width = thumb_width + 10
        cell_height = thumb_height + info_height + 10
        
        cols = max(1, (frame_width - 2 * margin) // cell_width)
        rows = max(1, (len(self.employee_images) + cols - 1) // cols)
        
        # Draw grid of employee images and data
        for i, name in enumerate(self.employee_images.keys()):
            if i >= rows * cols:  # Only show what fits on screen
                break
                
            row = i // cols
            col = i % cols
            
            x = margin + col * cell_width
            y = 80 + row * cell_height
            
            # Get employee image and data
            img = self.employee_images[name]
            data = self.employee_data[name]
            
            # Resize image to thumbnail size
            if img.shape[0] > 0 and img.shape[1] > 0:  # Make sure image is valid
                thumb = cv2.resize(img, (thumb_width, thumb_height))
                # Place thumbnail on screen
                screen[y:y+thumb_height, x:x+thumb_width] = thumb
            
            # Draw box around thumbnail
            cv2.rectangle(screen, (x, y), (x + thumb_width, y + thumb_height), 
                          self.ui_colors['text'], 1)
            
            # Add employee info
            y_text = y + thumb_height + 20
            cv2.putText(screen, f"NAME: {name}", (x, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['text'], 1)
            
            cv2.putText(screen, f"ID: {data['id']}", (x, y_text + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['text'], 1)
            
            cv2.putText(screen, f"CLEARANCE: {data['clearance']}", (x, y_text + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['text'], 1)
            
            # Add security badge style overlay
            clearance_color = {
                "LEVEL 1": (0, 255, 0),    # Green
                "LEVEL 2": (0, 255, 255),  # Yellow
                "LEVEL 3": (0, 165, 255),  # Orange
                "LEVEL 4": (0, 0, 255),    # Red
                "LEVEL 5": (128, 0, 128)   # Purple
            }.get(data['clearance'], (255, 255, 255))
            
            # Add small colored rectangle indicating clearance level
            cv2.rectangle(screen, (x + thumb_width - 30, y), (x + thumb_width, y + 10), 
                          clearance_color, -1)
        
        # Add database stats
        cv2.putText(screen, f"TOTAL PERSONNEL: {len(self.employee_images)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 1)
        
        return screen
    
    def run(self):
        """Run the secure authentication system."""
        # Initialize webcam
        print("Initializing secure system...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set webcam properties for better quality
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual frame size (may be different from requested)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Let the camera warm up
        time.sleep(1)
        
        print("Secure authentication system running.")
        print("Controls:")
        print("  - SPACE: Start facial scan")
        print("  - D: View personnel database (after authentication)")
        print("  - L or ESC: Lock system")
        print("  - R: Retry authentication")
        print("  - Q: Quit")
        
        # Variables for authentication process
        auth_start_time = 0
        auth_result = False
        identified_person = None
        
        # Create window with custom name
        window_name = "Secure Facility Authentication"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Grab a frame from camera
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break
            
            # Handle different states
            if self.state == "LOCKED":
                # Show the locked screen
                display = self.create_locked_screen(frame_width, frame_height)
                
            elif self.state == "SCANNING":
                # Update scan progress
                current_time = time.time()
                elapsed = current_time - auth_start_time
                scan_duration = 3.0  # seconds
                
                if elapsed < scan_duration:
                    # Update progress
                    self.scan_progress = elapsed / scan_duration
                    self.scan_line_pos = self.scan_progress
                    
                    # Display scanning screen
                    display = self.create_scanning_screen(frame)
                else:
                    # Scanning complete, perform facial recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        # Try to identify the first face
                        if len(face_encodings) > 0 and len(self.known_face_encodings) > 0:
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encodings[0])
                            
                            best_match_index = np.argmin(face_distances)
                            
                            if face_distances[best_match_index] < self.confidence_threshold:
                                # Authentication successful
                                auth_result = True
                                identified_person = self.known_face_names[best_match_index]
                                self.speak(f"Access granted. Welcome, {identified_person}")
                            else:
                                # Authentication failed
                                auth_result = False
                                self.speak("Access denied. Unauthorized personnel.")
                        else:
                            # No faces in database to compare to
                            auth_result = False
                            self.speak("Access denied. No authorized personnel in database.")
                    else:
                        # No face detected
                        auth_result = False
                        self.speak("Access denied. No face detected.")
                    
                    # Change state based on authentication result
                    if auth_result:
                        self.state = "ACCESS_GRANTED"
                    else:
                        self.state = "ACCESS_DENIED"
                    
                    # Reset progress
                    auth_start_time = current_time
            
            elif self.state == "ACCESS_GRANTED" or self.state == "ACCESS_DENIED":
                # Display the access result screen
                display = self.create_access_screen(frame, self.state == "ACCESS_GRANTED", identified_person)
                
                # Auto-transition to database if access granted (after delay)
                if self.state == "ACCESS_GRANTED":
                    current_time = time.time()
                    if (current_time - auth_start_time) > 3.0:
                        self.state = "DATABASE"
            
            elif self.state == "DATABASE":
                # Display the employee database screen
                display = self.create_database_screen(frame_width, frame_height)
            
            # Display the current screen
            cv2.imshow(window_name, display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
                
            elif key == ord(' ') and self.state == "LOCKED":
                # Start scanning
                self.state = "SCANNING"
                auth_start_time = time.time()
                self.scan_progress = 0
                self.scan_line_pos = 0
                self.speak("Initiating facial recognition scan")
                
            elif key == ord('l') or key == 27:  # 'l' or ESC
                # Lock system
                self.state = "LOCKED"
                self.speak("System locked")
                
            elif key == ord('r') and self.state in ["ACCESS_DENIED", "ACCESS_GRANTED"]:
                # Retry authentication
                self.state = "LOCKED"
                
            elif key == ord('d') and self.state == "ACCESS_GRANTED":
                # Show database
                self.state = "DATABASE"
                self.speak("Accessing personnel database")
        
        # Clean up
        video_capture.release()
        cv2.destroyAllWindows()
        print("Secure authentication system terminated.")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Secure Facility Authentication System')
    parser.add_argument('--database', '-d', default='faces_db', 
                        help='Path to the folder containing face images (default: faces_db)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Face matching confidence threshold (default: 0.6, lower is stricter)')
    
    args = parser.parse_args()
    
    # Create and run the authentication system
    auth_system = SecureAuthentication(args.database, args.threshold)
    auth_system.run()