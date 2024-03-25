import cv2 as cv
from easy_ViTPose import VitInference
import time, os, argparse
import numpy as np

VITPOSE_DATASET_WEIGHTS = "mpii"
# s (30 FPS), b (26 FPS), l (7 FPS), h (6 FPS)
VITPOSE_WEIGHTS = f"vitpose-l-{VITPOSE_DATASET_WEIGHTS}.pth"
YOLO_WEIGHTS_SIZE = "s"


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="ViTPose",
		description="Run ViTPose model",
		)
	
	parser.add_argument("--vitpose-size", 
		choices=["s", "b", "l", "h"], 
		dest="vitpose_size", 
		default="s",
		help="ViTPose model size")

	parser.add_argument("--yolo-size",
		choices=["s", "n"],
		dest="yolo_size",
		default="n",
		help="YOLOv8 size")

	parser.add_argument("--fps",
		dest="fps",
		action="store_true",
		default=False,
		help="Print FPS")

	args = parser.parse_args()
	VITPOSE_SIZE = args.vitpose_size

# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition
vitpose_weights_path = os.path.join("./vitpose_weights", VITPOSE_DATASET_WEIGHTS, f"vitpose-{VITPOSE_SIZE}-{VITPOSE_DATASET_WEIGHTS}.pth")
gt_vitpose_weights_path = os.path.join("./vitpose_weights", VITPOSE_DATASET_WEIGHTS, f"vitpose-h-{VITPOSE_DATASET_WEIGHTS}.pth")
yolo_weights_path = os.path.join("./yolo_weights", f"yolov8{YOLO_WEIGHTS_SIZE}.pt")

# If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
# If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
# dataset and det_class parameters can be inferred from the ckpt name, but you can specify them.
model = VitInference(vitpose_weights_path, yolo_weights_path, model_name=VITPOSE_SIZE, yolo_size=320, dataset=VITPOSE_DATASET_WEIGHTS, is_video=False, device='cuda')
gt_model = VitInference(gt_vitpose_weights_path, yolo_weights_path, model_name="h", yolo_size=320, dataset=VITPOSE_DATASET_WEIGHTS, is_video=False, device='cuda')




# init videocapture
cap = cv.VideoCapture(0)

# init variables for FPS computation
# new_time = prev_time = time.time()

while True:
	start_time = time.time()
	# obtain frame
	ret, frame = cap.read()

	# Image to run inference RGB format
	img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

	# if not ret:
	# 	print("No frame, exiting...")
	# 	break


	# Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
	# If is_video=True the IDs will be consistent among the ordered video frames.
	keypoints = model.inference(img)
	# gt_keypoints = gt_model.inference(img)
	# # compute FPS
	new_time = time.time()
	fps = 1/(new_time - start_time)
	start_time = new_time

	if args.fps:
		print(f"FPS: {fps}")
	# 25, 3: [Y, X, confidence]
	#print(model._keypoints)
	# for detection in model._keypoints:
	# 	print(f"Detection {detection}")
	# 	print(f"Confidence: {model._keypoints[detection][10, 2]}")
		# for Y, X, confidence in model._keypoints[detection]:
			# print(f"X: {X}, Y: {Y}, Confidence: {confidence}")
		# print("\n\n")
	# for point in model._keypoint[0]:
#		print(f"X: {point}")
	# call model.reset() after each video

	# img = gt_model.draw(img, show_yolo=True, confidence_threshold=0.5)

	# img = np.zeros(img.shape, dtype='uint8')

	#model._img = img
	img = model.draw(img, show_yolo=True, confidence_threshold=0.5)  # Returns RGB image with drawings

	
	
	# show frame
	cv.imshow("Camera", cv.cvtColor(img, cv.COLOR_RGB2BGR) )

	# exit when key 'q' is pressed
	if cv.waitKey(1) == ord('q'):
		break

# release videocapture
cap.release()
# close window
cv.destroyAllWindows()