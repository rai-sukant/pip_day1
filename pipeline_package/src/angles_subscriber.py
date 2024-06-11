#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray

def callback(data):
    rospy.loginfo("Received data: %s", data.data)

def listener():
    # Initialize the ROS node
    rospy.init_node('average_angles_listener', anonymous=True)
    
    # Subscribe to the 'average_angles' topic
    rospy.Subscriber("average_angles", Float32MultiArray, callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()