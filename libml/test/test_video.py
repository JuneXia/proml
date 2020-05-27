import os
import cv2

SAVE_PATH = '/home/xiajun/res/face/gcface/outdoor'
SAVE_FIELD = 'sunshine'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    count = 0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if not ret or 0xFF == ord('q'):
            break

        impath = os.path.join(SAVE_PATH, '{}-{:06n}.jpg'.format(SAVE_FIELD, count))
        cv2.imwrite(impath, frame)
        cv2.imshow('show', frame)
        cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()
    print('end!')