import numpy as np
import matplotlib.pyplot as plt
import cv2

rectangles = [cv2.imread("prostokat1.png"),
              cv2.imread("prostokat2.png"),
              cv2.imread("prostokat3.png"),
              cv2.imread("prostokat4.png"),
              cv2.imread("prostokat5.png")]
plt.figure(8)
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(rectangles[i])
    plt.title('Rectangular ' + str(i))

stars = [cv2.imread("gwiazda1.png"),
         cv2.imread("gwiazda2.png"),
         cv2.imread("gwiazda3.png"),
         cv2.imread("gwiazda4.png"),
         cv2.imread("gwiazda5.png")]
plt.figure(9)
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(stars[i])
    plt.title('Star ' + str(i))

huRec = np.zeros((7, 5))
huStar = np.zeros((7, 5))

for sta, rec, i in zip(stars, rectangles, range(5)):
    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
    sta = cv2.cvtColor(sta, cv2.COLOR_BGR2GRAY)
    huRec[0][i], huRec[1][i], huRec[2][i], huRec[3][i], huRec[4][i], huRec[5][i], huRec[6][i] = cv2.HuMoments(cv2.moments(rec)).flatten()
    huStar[0][i], huStar[1][i], huStar[2][i], huStar[3][i], huStar[4][i], huStar[5][i], huStar[6][i] = cv2.HuMoments(cv2.moments(sta)).flatten()

for i in range(7):
    plt.figure(i+1)
    plt.plot(range(5), huRec[i], 'o', range(5), huStar[i], 'o')
    plt.legend(("Rectangular", "Star"))
    plt.title('Hu Moment ' + str(i+1))
    plt.xlabel('Picture\'s number')

plt.show()
