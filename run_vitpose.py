import cv2 as cv
from easy_ViTPose import VitInference
import time, os, argparse
import numpy as np
from huggingface_hub import hf_hub_download

# s (30 FPS), b (26 FPS), l (7 FPS), h (6 FPS)


if __name__ == "__main__":
	
	# ARGUMENTS PARSER
	parser = argparse.ArgumentParser(
		prog="ViTPose",
		description="Run ViTPose model",
		)
	
	parser.add_argument("--vitpose-size", 
		choices=["s", "b", "l", "h"], 
		dest="vitpose_size", 
		default="s",
		help="ViTPose model size")

	parser.add_argument("--vitpose-dataset", 
		choices=["mpii", "coco_25", "wholebody"], 
		dest="vitpose_dataset", 
		default="coco_25",
		help="ViTPose dataset")

	parser.add_argument("--vitpose-conf-thr",  
		type=float,
		dest="vitpose_conf_thr", 
		default=0.5,
		help="ViTPose confidence threshold of keypoints detection")

	parser.add_argument("--joints-conf",
		dest="joints_conf",
		action="store_true",
		default=False,
		help="Show confidence score for each detected joint")

	parser.add_argument("--yolo-size",
		choices=["s", "n"],
		dest="yolo_size",
		default="n",
		help="YOLOv8 size")

	parser.add_argument("--yolo-conf-thr",
		type=float,
		dest="yolo_conf_thr",
		default=0.5,
		help="YOLOv8 confidence threshold of bbox detections")

	parser.add_argument("--fps",
		dest="fps",
		action="store_true",
		default=False,
		help="Print FPS")

	args = parser.parse_args()
	VITPOSE_SIZE = args.vitpose_size
	VITPOSE_DATASET = args.vitpose_dataset
	VITPOSE_WEIGHTS = f"vitpose-{VITPOSE_SIZE}-{VITPOSE_DATASET}.pth"
	YOLO_SIZE = args.yolo_size
	VITPOSE_WEIGHTS_PATH = os.path.join("./vitpose_weights", "torch", VITPOSE_DATASET, f"vitpose-{VITPOSE_SIZE}-{VITPOSE_DATASET}.pth")
	YOLO_WEIGHTS_PATH = os.path.join("./yolo_weights", "yolov8", f"yolov8{YOLO_SIZE}.pt")

	# DOWNLOAD WEIGHTS IF NOT EXIST
	if not os.path.exists("./vitpose_weights"):
		os.mkdir("./vitpose_weights")
	if not os.path.exists(VITPOSE_WEIGHTS_PATH):
		print(f"ViTPose weights not found locally! Downloading...")
		hf_hub_download(repo_id="JunkyByte/easy_ViTPose", 
			filename=f"torch/{VITPOSE_DATASET}/vitpose-{VITPOSE_SIZE}-{VITPOSE_DATASET}.pth", 
			local_dir="./vitpose_weights")

	if not os.path.exists("./yolo_weights"):
		os.mkdir("./yolo_weights")
	if not os.path.exists(YOLO_WEIGHTS_PATH):
		print(f"YOLOv8 weights not found locally! Downloading...")
		hf_hub_download(repo_id="JunkyByte/easy_ViTPose", 
			filename=f"yolov8/yolov8{YOLO_SIZE}.pt", 
			local_dir="./yolo_weights")

	# INIT VITPOSE MODEL FOR INFERENCE
	model = VitInference(VITPOSE_WEIGHTS_PATH, YOLO_WEIGHTS_PATH, model_name=VITPOSE_SIZE, yolo_size=320, dataset=VITPOSE_DATASET, is_video=False, device='cuda')


	# init videocapture
	cap = cv.VideoCapture(0)


	while True:
		start_time = time.time()
		# obtain frame
		ret, frame = cap.read()

		# Image to run inference RGB format
		img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


		# Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
		# If is_video=True the IDs will be consistent among the ordered video frames.
		keypoints = model.inference(img, yolo_conf_thr=args.yolo_conf_thr)
		
		# compute FPS
		new_time = time.time()
		fps = 1/(new_time - start_time)
		start_time = new_time

		# print FPS if requested
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

		# obtain image with pose estimations
		img = model.draw(show_yolo=True, vitpose_conf_thr=args.vitpose_conf_thr, draw_joints_confidence=args.joints_conf)  # Returns RGB image with drawings
	
		# show frame
		cv.imshow("Camera", cv.cvtColor(img, cv.COLOR_RGB2BGR) )

		# exit when key 'q' is pressed
		if cv.waitKey(1) == ord('q'):
			break

	# release videocapture
	cap.release()
	# close window
	cv.destroyAllWindows()