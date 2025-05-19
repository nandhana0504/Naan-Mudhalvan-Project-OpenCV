

import cv2
import numpy as np

# Generate and save ArUco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_img = cv2.aruco.generateImageMarker(aruco_dict, 23, 200)
cv2.imwrite("mark.jpg", marker_img)
print("Marker saved as mark.jpg")

# Read the saved marker image
image = cv2.imread("mark.jpg", cv2.IMREAD_COLOR)

# Check if image loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Marker detection setup
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

# Dummy camera calibration
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

# Draw and display
if ids is not None:
    print(f"Detected marker IDs: {ids.flatten()}")
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
    for rvec, tvec in zip(rvecs, tvecs):
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
else:
    print("No markers detected.")

# Show result
cv2.imshow('AR Marker Detection - Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()




