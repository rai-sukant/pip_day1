#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def callback_x(data):
    rospy.loginfo("Received contour_x: %d", data.data)

def callback_y(data):
    rospy.loginfo("Received contour_y: %d", data.data)

def listener():
    rospy.init_node('contour_listener', anonymous=True)

    rospy.Subscriber('contour_x', Int32, callback_x)
    rospy.Subscriber('contour_y', Int32, callback_y)

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
