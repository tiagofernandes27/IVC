import cv2

tracker = cv2.TrackerCSRT.create()

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

BB = cv2.selectROI(frame, False)
tracker.init(frame, BB)

while True:
    ret, frame = cap.read()
    track_success, BB = tracker.update(frame)
    if track_success:
        top_left = (int(BB[0]), int(BB[1]))
        bottom_right = (int(BB[0]+BB[2]), int(BB[1]+BB[3]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 5)
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
