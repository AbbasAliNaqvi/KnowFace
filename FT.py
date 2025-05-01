import cv2
import face_recognition
import numpy as np
import os
import time
import pyttsx3
from pathlib import Path

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
        

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  
        

        self.load_face_database()
        
    def load_face_database(self):
        """Load all face images from the database directory."""
        print(f"Loading face database from: {self.database_path}")
        

        image_extensions = ['.jpg', '.jpeg', '.png']

        for file in os.listdir(self.database_path):
            path = os.path.join(self.database_path, file)
            

            if os.path.isfile(path) and Path(file).suffix.lower() in image_extensions:

                name = os.path.splitext(file)[0]
                
                try:

                    image = face_recognition.load_image_file(path)
                    face_locations = face_recognition.face_locations(image)
                    
                    if face_locations:

                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                        

                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        
                        print(f"Added {name} to the database")
                    else:
                        print(f"No face found in {file}, skipping...")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        print(f"Database loaded with {len(self.known_face_names)} faces")
        

        self.speak(f"Database loaded with {len(self.known_face_names)} faces")
    
    def speak(self, text):
        """Use text-to-speech to speak the given text."""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def run(self):
        """Run the face recognition system with webcam input."""
        if not self.known_face_encodings:
            print("No faces in database. Please add images to the database folder.")
            self.speak("No faces in database. Please add images to the database folder.")
            return
        

        print("Starting webcam...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            self.speak("Could not open webcam.")
            return
        

        time.sleep(1)
        
        print("Face recognition system running. Press 'q' to quit.")
        self.speak("Face recognition system is now running")
        

        last_recognition_time = 0
        recognition_cooldown = 3 
        last_recognized_name = None
        
        while True:

            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break
            

            current_time = time.time()
            process_this_frame = (current_time - last_recognition_time) > recognition_cooldown
            

            display_frame = frame.copy()
            
            if process_this_frame:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                

                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:

                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    

                    best_match = {
                        "name": "Unknown",
                        "distance": 1.0,  
                        "location": None
                    }
                    

                    for face_encoding, face_location in zip(face_encodings, face_locations):

                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:

                            best_index = np.argmin(face_distances)
                            best_distance = face_distances[best_index]
                            

                            if best_distance < best_match["distance"]:
                                best_match["distance"] = best_distance
                                best_match["location"] = face_location
                                

                                if best_distance < self.confidence_threshold:
                                    best_match["name"] = self.known_face_names[best_index]
                    

                    if best_match["name"] != "Unknown" and best_match["location"] is not None:

                        last_recognition_time = current_time
                        

                        if best_match["name"] != last_recognized_name:
                            match_percentage = (1 - best_match["distance"]) * 100
                            print(f"Match found: {best_match['name']} ({match_percentage:.2f}%)")
                            self.speak(f"YOU KNOWN THIS PERSON  {best_match['name']}")
                            last_recognized_name = best_match["name"]
            

            for (top, right, bottom, left) in face_locations:

                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                

                name = best_match["name"] if best_match["location"] else "Processing..."
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            

            text = "Press 'q' to quit"
            cv2.putText(display_frame, text, (10, display_frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            

            cv2.imshow('Face Recognition System', display_frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

        video_capture.release()
        cv2.destroyAllWindows()
        print("Face recognition system stopped.")
        self.speak("Face recognition system stopped")

if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser(description='Face recognition system using a database of faces.')
    parser.add_argument('--database', '-d', default='faces_db', 
                        help='Path to the folder containing face images (default: faces_db)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Face matching confidence threshold (default: 0.6, lower is stricter)')
    
    args = parser.parse_args()
    
    # Create and runs the face recognition system
    face_system = FaceRecognitionSystem(args.database, args.threshold)
    face_system.run()