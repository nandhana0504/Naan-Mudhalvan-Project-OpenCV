import cv2
import numpy as np
genuine_note = cv2.imread("currency.jpg", 0)   
test_note = cv2.imread("fake currency.jpg", 0)       
genuine_color = cv2.cvtColor(genuine_note, cv2.COLOR_GRAY2BGR)
test_color = cv2.cvtColor(test_note, cv2.COLOR_GRAY2BGR)
cv2.putText(genuine_color, "Genuine Note", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(test_color, "Fake Note", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 255), 2, cv2.LINE_AA)
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(genuine_note, None)
kp2, des2 = orb.detectAndCompute(test_note, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matched_img = cv2.drawMatches(genuine_note, kp1, test_note, kp2, matches[:20], None, flags=2)
cv2.putText(matched_img, "Matches", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 255, 0), 2, cv2.LINE_AA)
good_matches = [m for m in matches if m.distance < 60]
match_percent = len(good_matches) / len(matches) * 100
print(f"Match Percentage: {match_percent:.2f}%")
if match_percent > 40:
    print("Result: The currency note is likely GENUINE.")
else:
    print("Result: The currency note is likely FAKE.")
cv2.imwrite("output_genuine_note_labeled.jpg", genuine_color)
cv2.imwrite("output_fake_note_labeled.jpg", test_color)
cv2.imwrite("output_matches_labeled.jpg", matched_img)
cv2.imshow("Genuine Note", genuine_color)
cv2.imshow("Fake Note", test_color)
cv2.imshow("Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
