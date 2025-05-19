import cv2
import numpy as np
background = cv2.imread("iphone.jpg")
background = cv2.resize(background, (640, 480))
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    mask = np.zeros(frame.shape[:2], np.uint8)
    rect = (50, 50, 600, 400) 
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = frame * mask_binary[:, :, np.newaxis]
    mask_inv = cv2.bitwise_not(mask_binary * 255)
    background_masked = cv2.bitwise_and(background, background, mask=mask_inv)
    final_output = cv2.add(foreground, background_masked)
    cv2.imshow("Virtual Background Replacement", final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
