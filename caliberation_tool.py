import cv2
import numpy as np

# --- UPDATE THIS BASED ON YOUR BOARD ---
# You printed 8x11 squares → 7x10 inner corners
pattern_size = (7, 10)  # (cols, rows of INNER corners)
square_size = 25.0      # mm (adjust if your printed square is different)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points for the chessboard pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # scale to real-world units (mm)

# Storage for points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Press SPACE to capture a frame when corners are detected.")
print("[INFO] Press ESC when done capturing.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret_corners:
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret_corners)
        cv2.putText(frame, "DETECTED - Press SPACE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NOT DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32 and ret_corners:  # SPACE
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"[INFO] Frame captured. Total: {len(objpoints)}")

cap.release()
cv2.destroyAllWindows()

# Run calibration if enough frames captured
if len(objpoints) > 5:
    print("[INFO] Running calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("[RESULT] Calibration successful!")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist.ravel())

    # Save calibration data
    np.savez("calib_data.npz", camera_matrix=mtx, dist_coeffs=dist)
    print("[INFO] Calibration data saved to calib_data.npz")
else:
    print("[ERROR] Not enough valid frames captured. Try again.")
