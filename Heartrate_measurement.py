import cv2
import numpy as np
import pywt
from scipy.signal import find_peaks
import time

# Constants for signal processing
FPS = 30  # Frames per second
WAVELET = 'db6'  # Wavelet type
SCALE = 2
HEART_RATE_INTERVAL = 10  # Update heart rate every 10 seconds
GAUSSIAN_BLUR_KERNEL = (3, 3)  # Smaller kernel size for faster blur
DEFAULT_HEART_RATE = 0  # Default heart rate value
MIN_BRIGHTNESS_THRESHOLD = 30  # Minimum brightness threshold for adaptive ROI
# Lower bound for skin tone in HSV
SKIN_TONE_LOWER = np.array([0, 20, 70], dtype="uint8")
# Upper bound for skin tone in HSV
SKIN_TONE_UPPER = np.array([30, 255, 255], dtype="uint8")


def extract_ppg_from_roi(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect skin tone
    skin_mask = cv2.inRange(hsv_frame, SKIN_TONE_LOWER, SKIN_TONE_UPPER)

    # Apply adaptive thresholding to find the brightest region within the skin tone mask
    _, thresh = cv2.threshold(
        skin_mask, MIN_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) > 0:
        # Get the largest contour (brightest region) as the ROI
        x, y, w, h = cv2.boundingRect(contours[0])
        roi = frame[y:y+h, x:x+w]

        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur with a smaller kernel to reduce noise (faster)
        gray_roi = cv2.GaussianBlur(gray_roi, GAUSSIAN_BLUR_KERNEL, 0)

        return gray_roi, (x, y, w, h)  # Extracted ROI and bounding box

    return None, None


def calculate_heart_rate(ppg_signal, fps):
    if ppg_signal is None:
        return None

    # Apply zero-padding to the PPG signal
    pad_length = len(ppg_signal) % (2 ** SCALE)
    if pad_length > 0:
        zero_padded_ppg_signal = np.pad(
            ppg_signal, (0, pad_length), 'constant')
    else:
        zero_padded_ppg_signal = ppg_signal

    # Perform wavelet decomposition on the zero-padded signal
    coeffs = pywt.wavedec(zero_padded_ppg_signal, WAVELET, level=SCALE)
    cA, *cD = coeffs  # Separate approximation and detail coefficients

    # Calculate the power spectrum for approximation coefficients
    power_spectrum = np.abs(cA) ** 2

    # Resize detail coefficients to have the same length as approximation coefficients
    for i in range(len(cD)):
        cD[i] = cv2.resize(cD[i], power_spectrum.shape[::-1],
                           interpolation=cv2.INTER_LINEAR)
        power_spectrum += np.abs(cD[i]) ** 2

    avg_power = np.mean(power_spectrum)

    # Sum along one axis to create a 1-D representation of the power spectrum
    power_spectrum_1d = np.sum(power_spectrum, axis=0)

    peaks, _ = find_peaks(power_spectrum_1d, distance=int(fps * 0.5))

    if len(peaks) < 2:
        return None

    # Calculate heart rate (beats per minute)
    bpm = 60 / (np.mean(np.diff(peaks)) / fps)
    return bpm


# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the camera window size
cap.set(3, 320)  # Width
cap.set(4, 240)  # Height

start_time = time.time()
heart_rate = DEFAULT_HEART_RATE  # Initialize heart rate to default value
update_time = start_time + HEART_RATE_INTERVAL

# Initialize object tracker
tracker = None
bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If tracker is not initialized, extract the PPG signal from the adaptive ROI
    if tracker is None:
        ppg_signal, bbox = extract_ppg_from_roi(frame)
    else:
        # Update the tracker with the current frame
        success, bbox = tracker.update(frame)
        if not success:
            # If tracking fails, reset the tracker and re-extract ROI
            tracker = None
            continue

    # If PPG signal is extracted and tracker is not initialized, initialize the tracker
    if ppg_signal is not None and tracker is None:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)

    # Calculate heart rate continuously
    heart_rate = calculate_heart_rate(ppg_signal, FPS)

    # Display the heart rate after the specified interval
    if time.time() >= update_time:
        if heart_rate is not None:
            print(f'Heart Rate: {heart_rate:.2f} BPM')
        update_time = time.time() + HEART_RATE_INTERVAL

    # Draw bounding box around the tracked object
    if bbox is not None:
        x, y, w, h = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Heart Rate Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()