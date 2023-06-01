import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # открываем камеру

mean = np.zeros(3)
std = np.zeros(3)
count = 0

while True:
    ret, frame = cap.read()  # читаем кадр с камеры
    if not ret:
        break

    mean += np.mean(frame, axis=(0, 1))
    std += np.std(frame, axis=(0, 1))
    count += 1

    cv2.imshow('frame', frame)  # показываем кадр на экране
    if cv2.waitKey(1) == ord('q'):  # выход при нажатии на клавишу q
        break

cap.release()
cv2.destroyAllWindows()

mean /= count
mean[2] -= 0.3 * 255
std /= count

print('Mean:', mean.astype(np.uint8)/255)
print('Std:', std.astype(np.uint8)/255)