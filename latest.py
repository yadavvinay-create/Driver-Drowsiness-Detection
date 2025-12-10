import dlib
import sys
import cv2  
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import simpleaudio as sa
import queue
from datetime import datetime
import os
import math


FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
EAR_THRESHOLD = 0.25  
MAR_THRESHOLD = 0.6   
EYE_AR_CONSEC_FRAMES = 6  
YAWN_CONSEC_FRAMES = 10   


modelPath = "models/shape_predictor_68_face_landmarks.dat"
cascade_path = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'models', 'haarcascade_frontalface_default.xml')
sound_path = "alarm.wav"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"[ERROR] Failed to load cascade from {cascade_path}")


LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
MOUTH_INDICES = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

class DrowsinessDetector:
    def _init_(self):
        self.eye_counter = 0
        self.yawn_counter = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.alarm_on = False
        self.thread_status_q = queue.Queue()
        
        
        self.current_ear = 0.0
        self.current_mar = 0.0
        
        
        self.session_start_time = datetime.now()
        self.session_ear_values = []
        self.session_mar_values = []
        self.session_alerts = 0
        
        
        self.ear_baseline = None
        self.calibration_frames = 0
        self.max_calibration_frames = 30
        
        print(f"Session started at: {self.session_start_time}")
        print("Calibrating... Please sit in normal awake position")

    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio - Using standard EAR calculation"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio - Using Colab version for better accuracy"""
        
        A = dist.euclidean(mouth[2], mouth[10])  
        B = dist.euclidean(mouth[4], mouth[8])   
        C = dist.euclidean(mouth[0], mouth[6])   
        mar = (A + B) / (2.0 * C)
        return mar

    def calibrate_baseline(self, ear):
        """Calibrate EAR baseline during initial frames"""
        if self.calibration_frames < self.max_calibration_frames:
            if self.ear_baseline is None:
                self.ear_baseline = ear
            else:
                
                self.ear_baseline = 0.9 * self.ear_baseline + 0.1 * ear
            self.calibration_frames += 1
            return False
        return True

    def get_dynamic_ear_threshold(self):
        """Get dynamic EAR threshold based on baseline"""
        if self.ear_baseline is None:
            return EAR_THRESHOLD
        
        return max(self.ear_baseline * 0.75, 0.20)  

    def check_eye_status(self, landmarks):
        """Check if eyes are open or closed"""
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
        
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        self.current_ear = ear
        self.session_ear_values.append(ear)
        
        
        calibration_done = self.calibrate_baseline(ear)
        
        if not calibration_done:
            return False, left_eye, right_eye, ear  
        
        dynamic_threshold = self.get_dynamic_ear_threshold()
        return ear < dynamic_threshold, left_eye, right_eye, ear

    def check_mouth_status(self, landmarks):
        """Check if mouth is open (yawning) - Improved detection"""
        mouth_points = [landmarks[i] for i in MOUTH_INDICES]
        
        mar = self.mouth_aspect_ratio(mouth_points)
        self.current_mar = mar
        self.session_mar_values.append(mar)
        
        
        return mar > MAR_THRESHOLD, mouth_points, mar

    def update_detection_counters(self, eyes_closed, yawning):
        """Update detection counters and trigger alerts - Improved logic"""
        
        if self.calibration_frames < self.max_calibration_frames:
            return

        
        if eyes_closed:
            self.eye_counter += 1
            if self.eye_counter >= EYE_AR_CONSEC_FRAMES and not self.alarm_on:
                self.trigger_alert("DROWSINESS ALERT: Prolonged eye closure detected!")
                self.blink_count += 1
        else:
            
            self.eye_counter = 0

        
        if yawning:
            self.yawn_counter += 1
            if self.yawn_counter >= YAWN_CONSEC_FRAMES and not self.alarm_on:
                self.trigger_alert("DROWSINESS ALERT: Yawning detected!")
                self.yawn_count += 1
        else:
            self.yawn_counter = 0

    def trigger_alert(self, message):
        """Trigger drowsiness alert"""
        print(f"ALERT: {message} - Time: {datetime.now().strftime('%H:%M:%S')}")
        self.session_alerts += 1
        self.alarm_on = True
        
        
        self.thread_status_q.put(False)
        thread = Thread(target=self.sound_alert, args=(sound_path, self.thread_status_q))
        thread.setDaemon(True)
        thread.start()

    def sound_alert(self, path, thread_status_q):
        """Play sound alert in separate thread"""
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
        except Exception as e:
            print(f"Error loading sound file: {e}")
            return

        while True:
            if not thread_status_q.empty():
                finished = thread_status_q.get()
                if finished:
                    break
            
            try:
                play_obj = wave_obj.play()
                play_obj.wait_done()
            except Exception as e:
                print(f"Error playing sound: {e}")
                break
            
            time.sleep(1.0)

    def reset_alarm(self):
        """Reset alarm state"""
        self.alarm_on = False
        self.thread_status_q.put(True)
        
        self.eye_counter = 0
        self.yawn_counter = 0

    def get_landmarks(self, image):
        """Extract facial landmarks from image"""
        
        small_image = cv2.resize(image, None, 
                               fx=1.0/FACE_DOWNSAMPLE_RATIO, 
                               fy=1.0/FACE_DOWNSAMPLE_RATIO, 
                               interpolation=cv2.INTER_LINEAR)

        faces = detector(small_image, 0)
        if len(faces) == 0:
            return None

        
        face = faces[0]
        original_rect = dlib.rectangle(
            int(face.left() * FACE_DOWNSAMPLE_RATIO),
            int(face.top() * FACE_DOWNSAMPLE_RATIO),
            int(face.right() * FACE_DOWNSAMPLE_RATIO),
            int(face.bottom() * FACE_DOWNSAMPLE_RATIO)
        )

        landmarks = predictor(image, original_rect)
        return [(p.x, p.y) for p in landmarks.parts()]

    def detect_faces(self, frame):
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        return len(faces)

    def draw_landmarks(self, frame, left_eye, right_eye, mouth):
        """Draw facial landmarks on frame"""
    
        for point in left_eye:
            cv2.circle(frame, tuple(map(int, point)), 2, (0, 0, 255), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(map(int, point)), 2, (0, 0, 255), -1)
        
        
        mouth_color = (255, 0, 0)  
        if self.current_mar > MAR_THRESHOLD:
            mouth_color = (0, 0, 255)  
            
        for point in mouth:
            cv2.circle(frame, tuple(map(int, point)), 2, mouth_color, -1)

    def display_metrics(self, frame, faces_detected):
        """Display detection metrics on frame"""
        y_offset = 30
        line_height = 25
        
        
        if self.calibration_frames < self.max_calibration_frames:
            calibration_progress = (self.calibration_frames / self.max_calibration_frames) * 100
            cv2.putText(frame, f"Calibrating: {calibration_progress:.0f}%", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
        else:
            cv2.putText(frame, "Calibration Complete", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
        
        metrics = [
            f"Faces: {faces_detected}",
            f"Blinks: {self.blink_count}",
            f"Yawns: {self.yawn_count}",
            f"EAR: {self.current_ear:.3f} (thresh: {self.get_dynamic_ear_threshold():.3f})",
            f"MAR: {self.current_mar:.3f} (thresh: {MAR_THRESHOLD})",
            
            f"Alerts: {self.session_alerts}"
        ]
        
        for i, metric in enumerate(metrics):
            color = (0, 255, 255)  
            
            # Color coding for better visibility
            if "EAR" in metric and self.current_ear < self.get_dynamic_ear_threshold():
                color = (0, 0, 255)  # Red when eyes closed
            elif "MAR" in metric and self.current_mar > MAR_THRESHOLD:
                color = (0, 165, 255)  # Orange when yawning
            elif "Alert" in metric and self.session_alerts > 0:
                color = (0, 0, 255)  # Red for alerts
                
            cv2.putText(frame, metric, (10, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def display_alerts(self, frame, eyes_closed, yawning):
        """Display alert messages on frame"""
        if self.calibration_frames < self.max_calibration_frames:
            return
            
        alerts = []
        y_offset = 300
        
        if self.eye_counter >= EYE_AR_CONSEC_FRAMES:
            alerts.append("! ! ! EYES CLOSED ALERT ! ! !")
        if self.yawn_counter >= YAWN_CONSEC_FRAMES:
            alerts.append("! ! ! YAWNING ALERT ! ! !")
        
        for i, alert in enumerate(alerts):
            cv2.putText(frame, alert, (50, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def display_yawn_status(self, frame, yawning):
        """Display clear yawn status"""
        if yawning:
            cv2.putText(frame, "YAWNING DETECTED!", (200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def get_session_summary(self):
        """Get session statistics"""
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds() / 60
        
        avg_ear = np.mean(self.session_ear_values) if self.session_ear_values else 0
        avg_mar = np.mean(self.session_mar_values) if self.session_mar_values else 0
        
        summary = {
            "duration_minutes": round(duration, 2),
            "total_frames": len(self.session_ear_values),
            "average_ear": round(avg_ear, 4),
            "average_mar": round(avg_mar, 4),
            "alerts_triggered": self.session_alerts,
            "blinks_detected": self.blink_count,
            "yawns_detected": self.yawn_count,
            "ear_baseline": round(self.ear_baseline, 4) if self.ear_baseline else 0,
            "start_time": self.session_start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        return summary

    def print_session_summary(self):
        """Print session summary to console"""
        summary = self.get_session_summary()
        
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {summary['duration_minutes']} minutes")
        print(f"Frames processed: {summary['total_frames']}")
        print(f"Average EAR: {summary['average_ear']}")
        print(f"Average MAR: {summary['average_mar']}")
        print(f"EAR Baseline: {summary['ear_baseline']}")
        print(f"Alerts triggered: {summary['alerts_triggered']}")
        print(f"Blinks detected: {summary['blinks_detected']}")
        print(f"Yawns detected: {summary['yawns_detected']}")
        print("="*50)

def main():
    """Main function to run drowsiness detection"""
    
    detector = DrowsinessDetector()
    
    
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Starting drowsiness detection...")
    print("Please sit in normal position for calibration (30 frames)")
    print("Press 'r' to reset alarms, 'q' to quit")
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            
            height, width = frame.shape[:2]
            resize_factor = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None, 
                             fx=1/resize_factor, fy=1/resize_factor,
                             interpolation=cv2.INTER_LINEAR)
            
            
            faces_detected = detector.detect_faces(frame)
            
            
            landmarks = detector.get_landmarks(frame)
            
            eyes_closed = False
            yawning = False
            
            if landmarks is not None:
                
                eyes_closed, left_eye, right_eye, ear = detector.check_eye_status(landmarks)
                
                
                yawning, mouth_points, mar = detector.check_mouth_status(landmarks)
                
                
                if detector.calibration_frames >= detector.max_calibration_frames:
                    detector.update_detection_counters(eyes_closed, yawning)
                
                
                detector.draw_landmarks(frame, left_eye, right_eye, mouth_points)
                
                detector.display_yawn_status(frame, yawning)
            
            
            detector.display_metrics(frame, faces_detected)
            detector.display_alerts(frame, eyes_closed, yawning)
            
            
            cv2.imshow("Drowsiness Detection - Eye & Yawn Monitoring", frame)
            
        
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_alarm()
                print("Alarms reset")
            elif key == ord('c'):
                # Force recalibration
                detector.calibration_frames = 0
                detector.ear_baseline = None
                print("Recalibration started...")
    
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        
        capture.release()
        cv2.destroyAllWindows()
        detector.print_session_summary()

if _name_ == "_main_":
    main()