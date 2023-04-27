import cv2

stream_url = 'http://127.0.0.1:5000/video_feed'

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()