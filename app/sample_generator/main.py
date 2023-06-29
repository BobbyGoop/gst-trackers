#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
from pathlib import Path

# import click
import cv2
import numpy as np
from tqdm import tqdm

from app.sample_generator.CustomConfig import CustomConfig
from app.sample_generator.MozaicSquare import MosaicSquare
from app.sample_generator.Obstacle import Obstacle

# FPS = 30

# BG_COLOR = 128
# PREROLL_FRAMES = FPS

cfg = CustomConfig.load_json('config.json')
_VID = cfg.video
_OBJ = cfg.object
_BAR = cfg.obstacle

IMG_EXT = ".jpg"
VID_EXT = ".mp4"

OUT_DIR = f"../video/{_VID.dir_name}/"

def main():
	"""
		- Размер объекта - есть
		- Скорость изменения размера объекта - есть
		- Скорость движения - есть
		- Контрастность объекта относительно фона - непонятно
		- Уровень шумов - есть
		- Размер препятствия (для моделирования перекрытий) - не понятно
		- Количество одновременно движущихся объектов (для моделирования перескока на соседний объект).
	"""

	# with open('config.json') as file:
	#     cfg = json.load(file)

	# Video params
	# width = cfg.video.width
	# height = cfg.video.height
	# output = cfg.video.output
	# noise_level = cfg.video.noise

	# grayscale = cfg.video.grayscale
	# FPS = cfg["video"]["framerate"] if cfg["video"]["framerate"] is not None else FPS
	# print(FPS)
	# BG_COLOR = cfg["video"]["bg_color"] if cfg["video"]["bg_color"]  is not None else BG_COLOR
	# duration = cfg["video"]["duration"]
	#
	# # Object params
	# object_size = cfg["object"]["size"]
	# object_speed = cfg["object"]["speed"]
	# object_pulsation = cfg["object"]["pulsation"]
	# object_contrast = cfg["object"]["size"]
	#
	# # Obstacle params
	# obstacle_size = cfg["obstacle"]["size"]
	# obstacle_number = cfg["obstacle"]["number"]
	# obstacle_color = cfg["obstacle"]["bg_color"]

	# if len(os.listdir(IMG_FOLDER)) == 0:
	# BORDER_SIZE = 2
	# object_coords = [width // 2 - object_size // 2 - int(object_size * 0.75),
	#                     height // 2 - object_size // 2 - int(object_size * 0.75)]
	#
	# borders = [[BORDER_SIZE, BORDER_SIZE],
	#         [width - BORDER_SIZE, height - BORDER_SIZE]]

	# angle = np.random.uniform(-np.pi, np.pi)
	# axial_speeds = np.array([np.cos(angle), np.sin(angle)]) * object_speed

	# out_dir = Path(output)
	# if out_dir.is_dir():
	#     raise FileExistsError(f"{out_dir} directory already exists")
	# out_dir.mkdir()

	tempdir = Path(OUT_DIR)
	# Create directory if it does not exist
	if not tempdir.is_dir():
		tempdir.mkdir()
	# Else delete all files
	else:
		for root, dirs, files in os.walk(tempdir):
			for f in files:
				os.unlink(os.path.join(root, f))
	# if not tempdir.is_dir():
	# rmtree(tempdir)

	frames_number = _VID.duration * _VID.framerate
	print(f"Generating {frames_number} images...")

	# if _BAR.alpha_enabled:
	# 	add_img_path = "./tree_branch.png"
	# else:
	# 	add_img_path = "./brick_wall.jpg"
	add_img_path = "assets/tree_branch.png"
	if _BAR.enabled:
		obst = Obstacle(os.path.abspath(add_img_path), _VID.width, _VID.height, _BAR.size)
	else:
		obst = None

	# offset = _BAR.size
	offset = _OBJ.center_offset
	print(offset)
	objects = []
	for i in range(_OBJ.number):
		objects.append(
			MosaicSquare(
				_VID.width,
				_VID.height,
				_OBJ.size,
				_OBJ.pulsation,
				_VID.border,
				_OBJ.speed,
				dx=offset,
				dy=0,
				CM=_OBJ.contrast_mode.enabled,
			)
		)
		# offset += _OBJ.size

	if _OBJ.contrast_mode.enabled:
		br = _OBJ.contrast_mode.color
		bg = _VID.bg_color
		print("Relative contrast: ", abs((br - bg) / (br + bg)))
	# title = "Image Generator"
	# cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
	# cv2.resizeWindow(title, _VID.width, _VID.height)
	overlapped_frames = []
	with tqdm(total=frames_number) as bar:
		for number in range(frames_number):
			# Generate background
			img = np.zeros((_VID.height, _VID.width, 3)) + _VID.bg_color

			coords_sq = []
			# Add objects
			for sq in objects:
				img = sq.draw_rectangle(img, number, _VID.framerate, _OBJ.contrast_mode.color)
				coords_sq.append(sq.get_opposite_coords())

			# if _BAR.alpha_enabled:
			# 	img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2BGRA)
			# Add obstacle
			if obst:
				img = obst.draw_obst(img)
				if len(coords_sq) == 1:
					res = overlap(coords_sq.pop(), obst.get_opposite_coords())
					# print(res, frames_number)
				# for c in coords_sq:
				# 	if overlap(c, obst.get_opposite_coords()):
				# 		overlapped_frames.append(number)
				# 		print("Overlapped: ", number)
				# 		print(number)
				# 		print(c)
				# 		print(obst.get_opposite_coords())
			# img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
			# Add noise
			if _VID.noise > 0:
				img += np.random.normal(0, _VID.noise, img.shape)

			# img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGRA2BGR)
			# Convert to grayscale
			if _VID.grayscale or _OBJ.contrast_mode.enabled:
				img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
			# else:
			# 	img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
			cv2.imwrite(f'{OUT_DIR}/{number:05d}{IMG_EXT}', img)
			# if number == 400:
			# 	break
			bar.update(1)
	# else:
	#     print("Directory is not empty. Using existing images...")

	# Writing video
	cmd = subprocess.run(fr"ffmpeg -framerate {_VID.framerate} -i {OUT_DIR}/%05d{IMG_EXT} -c:v libx264 {OUT_DIR}/{_VID.dir_name}.{VID_EXT} -y", check=True)
	cmd.check_returncode()

	# video = write_video_ffmpeg(input=tempdir, output=_VID.output)
	print(f"Output video: {_VID.dir_name}.{VID_EXT}")
	# show_result(video)


def show_result(video_path):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	cap = cv2.VideoCapture(video_path)

	# Check if camera opened successfully
	if not cap.isOpened():
		print("Error opening video stream or file")

	# Read until video is completed
	while cap.isOpened():
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret:
			# Display the resulting frame
			cv2.imshow('Frame', frame)
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		# Break the loop
		else:
			break

	# When everything done, release the video capture object
	cap.release()
	# Closes all the frames
	cv2.destroyAllWindows()

# Returns true if two rectangles(l1, r1)
# and (l2, r2) overlap
def overlap(square, wall):
	# print("  Frame: ", number)
	# print(" Object: ", [square.x1, square.y1], [square.x2, square.y2])
	# print("Barrier: ", [wall.x1, wall.y1], [wall.x2, wall.y2])

	# square=cv2.rectangle()
	x_match = False
	y_match = False
	if (square.x1 < wall.x2) and (square.x1 > wall.x1 or square.x2 > wall.x1):  # Right to Left
		x_match = True
	if (square.x2 > wall.x1) and (square.x1 < wall.x2 or square.x2 < wall.x2):  # Left to Right
		x_match = True

	if (square.y2 > wall.y1) and (square.y1 < wall.y2 or square.y2 < wall.y2):  # Up to down
		y_match = True
	if (square.y1 < wall.y2) and (square.y1 > wall.y1 or square.y2 > wall.y2):  # Down Up
		y_match = True
	# else:
	# 	y_match = False
	# if (wall.y2 > square.y1 and wall.y2 < square.y2) or (wall.y1 > square.y1 and wall.y1 < square.y2):
	# 	y_match = True
	# else:
	# 	y_match = False
	# print("X match / Y match: ", x_match, y_match)
	# print(x_match, y_match, number)
	# if x_match and y_match:
	# 	return True
	# else:
	# 	return False
	return x_match and y_match


def will_bound(min_coord, max_coord, coord, speed, size):
	next_coord = coord + speed
	return next_coord + size > max_coord or next_coord < min_coord


def calculate_next_position(borders, coords, speeds, size):
	start_point = coords.copy()
	for i, _ in enumerate(coords):
		if will_bound(borders[0][i], borders[1][i], coords[i], speeds[i], size):
			speeds[i] *= -1
		start_point[i] = start_point[i] + speeds[i]

	return start_point


def calculate_size(amplitude, frequency, base_size, frame_number):
	phase = amplitude * np.sin(frame_number * 2 * np.pi * frequency / _VID.framerate)
	if phase >= 0:
		phase += 1
		res = base_size * phase
	else:
		phase -= 1
		res = base_size / phase
	return abs(res)


def obstacle_rectangle(image, center_coords, size):
	start_point = (center_coords[0] - size // 2, center_coords[1] - size // 2)
	end_point = (center_coords[0] + size // 2, center_coords[1] + size // 2)
	color = (255, 0, 0)
	thickness = 2
	image = cv2.rectangle(image, start_point, end_point, color, thickness)

	return image


def mosaic_rectangle(image, start, end, n=2):
	x0, y0 = start
	xn, yn = end

	X = np.linspace(x0, xn, n + 1, dtype=int)
	Y = np.linspace(y0, yn, n + 1, dtype=int)

	colors = {1: (128, 255, 255), -1: (0, 0, 64)}
	idx = 1

	for i in range(n):
		for j in range(n):
			color = colors[idx]
			image = cv2.rectangle(image, (X[i], Y[j]), (X[i + 1], Y[j + 1]), color, -1)
			idx *= -1
		idx *= -1

	return image


def write_video(input, output):
	images = [img for img in os.listdir(input) if img.endswith(IMG_EXT)]
	frame = cv2.imread(os.path.join(input, images[0]))
	height, width, layers = frame.shape

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	video = cv2.VideoWriter(output, fourcc, _VID.framerate, (width, height))
	for image in images:
		video.write(cv2.imread(os.path.join(input, image)))

	cv2.destroyAllWindows()
	video.release()

	return output


def write_video_ffmpeg(input, output):

	return output


if __name__ == "__main__":
	main()
