# %%
import math
from datetime import datetime
import time
import numpy as np
import mediapipe as mp
import cv2

# %%


class SmileDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Landmark indices for mouth corners and lips
        self.LEFT_MOUTH_CORNER = 61
        self.RIGHT_MOUTH_CORNER = 291
        self.TOP_LIP_CENTER = 13
        self.BOTTOM_LIP_CENTER = 14
        self.UPPER_LIP_TOP = 12
        self.LOWER_LIP_BOTTOM = 15

        # Detection parameters
        self.smile_threshold = 0.0025  # Adjust based on testing
        self.min_smile_duration = 0.01  # Minimum smile duration (0.01 seconds)
        self.recording_duration = 10.0  # Total recording time (10 seconds)
        self.smile_start_time = None
        self.recording_start_time = None
        self.smile_captured = False

    def calculate_mouth_aspect_ratio(self, landmarks, image_shape):
        """Calculate mouth aspect ratio to detect smiles"""
        h, w = image_shape[:2]

        # Get mouth corner coordinates
        left_corner = landmarks[self.LEFT_MOUTH_CORNER]
        right_corner = landmarks[self.RIGHT_MOUTH_CORNER]
        top_lip = landmarks[self.TOP_LIP_CENTER]
        bottom_lip = landmarks[self.BOTTOM_LIP_CENTER]

        # Convert normalized coordinates to pixel coordinates
        left_corner_px = (int(left_corner.x * w), int(left_corner.y * h))
        right_corner_px = (int(right_corner.x * w), int(right_corner.y * h))
        top_lip_px = (int(top_lip.x * w), int(top_lip.y * h))
        bottom_lip_px = (int(bottom_lip.x * w), int(bottom_lip.y * h))

        # Calculate distances
        mouth_width = math.sqrt((right_corner_px[0] - left_corner_px[0])**2 +
                                (right_corner_px[1] - left_corner_px[1])**2)
        mouth_height = math.sqrt((top_lip_px[0] - bottom_lip_px[0])**2 +
                                 (top_lip_px[1] - bottom_lip_px[1])**2)

        # Calculate mouth aspect ratio
        if mouth_height > 0:
            mar = mouth_width / mouth_height
        else:
            mar = 0

        return mar

    def calculate_mouth_curvature(self, landmarks, image_shape):
        """Calculate mouth curvature to detect smiles"""
        h, w = image_shape[:2]

        # Get key mouth points
        left_corner = landmarks[self.LEFT_MOUTH_CORNER]
        right_corner = landmarks[self.RIGHT_MOUTH_CORNER]
        top_lip = landmarks[self.UPPER_LIP_TOP]

        # Convert to pixel coordinates
        left_px = (left_corner.x * w, left_corner.y * h)
        right_px = (right_corner.x * w, right_corner.y * h)
        top_px = (top_lip.x * w, top_lip.y * h)

        # Calculate the angle of mouth curvature
        # If mouth corners are higher than the center, it's likely a smile
        mouth_center_y = (left_px[1] + right_px[1]) / 2
        # Normalize by image width
        curvature = (top_px[1] - mouth_center_y) / w

        return curvature

    def is_smiling(self, landmarks, image_shape):
        """Determine if the person is smiling based on facial landmarks"""
        try:
            mar = self.calculate_mouth_aspect_ratio(landmarks, image_shape)
            curvature = self.calculate_mouth_curvature(landmarks, image_shape)

            # A smile typically has:
            # 1. Higher mouth aspect ratio (wider mouth)
            # 2. Positive curvature (mouth corners higher than center)
            smile_detected = mar > 3.0 and curvature > self.smile_threshold

            return smile_detected, mar, curvature
        except Exception as e:
            print(f"Error in smile detection: {e}")
            return False, 0, 0

    def save_image(self, frame):
        """Save the captured frame with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"smile_capture_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image saved as: {filename}")
        return filename

    def run(self):
        """Main application loop"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Smile Detection Application Started!")
        print("Recording for 10 seconds. Smile to capture an image!")
        print("Press 'q' to quit.")

        # Start recording timer
        self.recording_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            current_time = time.time()

            # Check if recording time is up
            elapsed_recording_time = current_time - self.recording_start_time
            if elapsed_recording_time >= self.recording_duration:
                if self.smile_captured:
                    print("Recording complete! Image was captured during the session.")
                else:
                    print(
                        "Recording complete! No smile was detected during the 10-second session.")
                break

            # Calculate remaining recording time
            remaining_time = self.recording_duration - elapsed_recording_time

            # Process the frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)

            smile_detected = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Check if person is smiling
                    smile_detected, mar, curvature = self.is_smiling(
                        face_landmarks.landmark, frame.shape
                    )

                    # Draw mouth landmarks for visualization
                    h, w = frame.shape[:2]
                    mouth_points = [self.LEFT_MOUTH_CORNER, self.RIGHT_MOUTH_CORNER,
                                    self.TOP_LIP_CENTER, self.BOTTOM_LIP_CENTER]

                    for point_idx in mouth_points:
                        point = face_landmarks.landmark[point_idx]
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    # Display detection values
                    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Curvature: {curvature:.4f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Handle smile detection and capture
            if smile_detected and not self.smile_captured:
                if self.smile_start_time is None:
                    self.smile_start_time = current_time

                smile_duration = current_time - self.smile_start_time

                # Check if smile has been maintained for minimum duration
                if smile_duration >= self.min_smile_duration:
                    print(
                        f"Smile detected for {smile_duration:.3f} seconds! Capturing image...")
                    filename = self.save_image(frame)
                    print(f"Success! Image captured and saved as {filename}")
                    self.smile_captured = True

                # Display smile detection status
                cv2.putText(frame, "SMILING DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Reset smile timer if not smiling
                self.smile_start_time = None

                if not self.smile_captured:
                    cv2.putText(frame, "Smile to capture image", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Image captured! Recording continues...", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Display recording timer
            cv2.putText(frame, f"Recording: {remaining_time:.1f}s remaining",
                        (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame
            cv2.imshow('Smile Detector', frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Application terminated by user.")
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the application"""
    try:
        detector = SmileDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

# %%
if __name__ == "__main__":
    main()
# %%
