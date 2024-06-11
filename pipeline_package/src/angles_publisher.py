#!/usr/bin/env python


import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

def CLAHE(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray_image)
    if len(image.shape) == 3:
        clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    return clahe_image

def white_balance(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    balanced_lab_image = cv2.merge((cl, a, b))
    balanced_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2BGR)
    return balanced_image

def Contrast_Up(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    return contrasted_image

def Contrast_Down(image):
    contrasted_image = cv2.convertScaleAbs(image, alpha=0.2, beta=0)
    return contrasted_image

def Brightness_Up(image):
    brightened_image = cv2.convertScaleAbs(image, alpha=2, beta=150)
    return brightened_image

def Brightness_Down(image):
    darkened_image = cv2.convertScaleAbs(image, alpha=1.0, beta=10)
    return darkened_image

def Masking(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 50])
    upper_bound = np.array([60, 255, 255])
    mask = cv2.inRange(image, lower_bound, upper_bound)
    new_img = cv2.bitwise_and(image, image, mask=mask)
    new_img = cv2.GaussianBlur(new_img, (55, 55), 0)
    kernel = np.array([[-1,-1,-1], [-1,99,-1], [-1,-1,-1]])
    new_img = cv2.filter2D(new_img, -1, kernel)
    new_img = cv2.bilateralFilter(new_img, 9, 1000, 1000)
    return new_img

def Make_Hough_Lines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_angle_deg = None
    average_angle = None
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold = 220
    edges = np.uint8(gradient_magnitude > threshold) * 255
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = cv2.convertScaleAbs(edges)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angles.append(theta)
        average_angle = np.mean(angles)
        average_angle_deg = np.degrees(average_angle)
    if average_angle_deg:
        return image, average_angle_deg
    else:
        return image, None

def main():
    rospy.init_node('angle_publisher', anonymous=True)
    pub = rospy.Publisher('average_angles', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    video_feed = cv2.VideoCapture('/home/sukant/pipes.mp4')

    while not rospy.is_shutdown():
        ret, frame = video_feed.read()
        if not ret:
            break

        height, width, _ = frame.shape
        part_height = height // 3

        part1 = frame[0:part_height, :]
        part2 = frame[part_height:2*part_height, :]
        part3 = frame[2*part_height:, :]

        n_part1, avg1 = Make_Hough_Lines(Masking(part1))
        n_part2, avg2 = Make_Hough_Lines(Masking(part2))
        n_part3, avg3 = Make_Hough_Lines(Masking(part3))

        cv2.imshow("Part1", part1)
        cv2.imshow("Part2", part2)
        cv2.imshow("Part3", part3)
        cv2.imshow("orig Part1", n_part1)
        cv2.imshow("orig Part2", n_part2)
        cv2.imshow("orig Part3", n_part3)

        if avg1 and avg2 and avg3:
            print(avg1)
            print(avg2)
            print(avg3)
            print("\n\n\n")
            angles_msg = Float32MultiArray(data=[avg1, avg2, avg3])
            pub.publish(angles_msg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()

    video_feed.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
