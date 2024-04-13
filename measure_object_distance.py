
import cv2
from realsense_camera import *
from mask_rcnn import *



rs = RealsenseCamera()
mrcnn = MaskRCNN()
#import os
#os.environ['QT_PLUGIN_PATH'] = '/home/saifhs/.local/lib/python3.10/site-packages/cv2/qt/plugins'
#os.environ["QT_QPA_PLATFORM"] = "wayland"
#import os 
#os.environ['QT_QPA_PLATFORM'] = 'xcb'
# Load Realsense camera


while True:
	# Get frame in real time from Realsense camera
	ret, bgr_frame, depth_frame = rs.get_frame_stream()
	if ret:
		print("Dimensions of bgr_frame:", bgr_frame.shape)
		cv2.imshow("Bgr frame", bgr_frame)
	else:
		print("Failed to capture bgr_frame")
	# Get object mask
	boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)
	
	# Draw object mask
	bgr_frame = mrcnn.draw_object_mask(bgr_frame)
	
	# Show depth info of the objects
	mrcnn.draw_object_info(bgr_frame, depth_frame)
	

	cv2.imshow("depth frame", depth_frame)
	cv2.imshow("Bgr frame", bgr_frame)
	
	key = cv2.waitKey(1)
	if key == 27:
		break

rs.release()
cv2.destroyAllWindows()
