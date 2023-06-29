import numpy as np
import cv2
from collections import namedtuple

class MosaicSquare:
	def __init__(self, frame_width, frame_height, size, pulsation, border, speed, dx=0, dy=0, hor=True, ver=True, base=True, colors = None):

		self.frame_width, self.frame_height = frame_width, frame_height
		# self.contrast_mode = CM
		self.amplitude = pulsation
		self.borders = [[border, border], [self.frame_width - border, self.frame_height - border]]
		# if hor and ver:
			# angle = np.random.uniform(-np.pi, np.pi)
			# angle = np.pi / 4

			# print(angle)

		if hor and not ver:
			self.axial_speeds = [speed, 0]
		elif ver and not hor:
			self.axial_speeds = [0, speed]
		# self.speed = speed

		self.base_size, self.new_size = size, size
		# Rectangle in center when initialized
		self.start_coords = [
			self.frame_width // 2 - self.base_size // 2 - dx,
			self.frame_height // 2 - self.base_size // 2 - dy
		]
		self.coords_tuple = namedtuple('CoordsTuple', 'x1 x2 y1 y2')

		if base:
			angle = np.pi / 3
			# BLUE GREEN RED ALPHA
			self.color_primary = (128, 255, 255, 255)
			self.color_secondary = (0, 0, 64, 255)
		else:
			angle = np.random.uniform(-np.pi, np.pi)
			if not colors:
				R = np.random.randint(0, 255)
				G = np.random.randint(0, 255)
				B = np.random.randint(0, 255)
				# BLUE GREEN RED ALPHA
				self.color_primary = (B, G, R, 255)
				self.color_secondary = (R, G, B, 255)
			else:
				self.color_primary = (colors[0][0],colors[0][1], colors[0][2],colors[0][3])
				self.color_secondary = (colors[1][0], colors[1][1], colors[1][2], colors[1][3])
			# self.color_secondary = self.color_primary

		self.axial_speeds = np.array([np.cos(angle), np.sin(angle)]) * speed
		# color_bg = (128, 128, 128, 128)
		#
		# # Convert to [0; 1]
		# color_prim = np.asarray(self.color_primary[0:3], dtype=np.float32) / 255
		# color_sec = np.asarray(self.color_secondary[0:3], dtype=np.float32) / 255
		# color_bg = np.asarray(color_bg[0:3], dtype=np.float32) / 255
		#
		# # Linerising
		# color_lin_prim = [i / 12.92 if i <= 0.04045 else ((i + 0.055) / 1.055) ** 2.4 for i in color_prim]
		# color_lin_sec = [i / 12.92 if i <= 0.04045 else ((i + 0.055) / 1.055) ** 2.4 for i in color_sec]
		# color_lin_bg = [i / 12.92 if i <= 0.04045 else ((i + 0.055) / 1.055) ** 2.4 for i in color_bg]
		#
		# Y_prim = 0.2126 * color_lin_prim[2] + 0.7152 * color_lin_prim[1] + 0.0722 * color_lin_prim[0]
		# Y_sec = 0.2126 * color_lin_sec[2] + 0.7152 * color_lin_sec[1] + 0.0722 * color_lin_sec[0]
		# Y_bg = 0.2126 * color_lin_bg[2] + 0.7152 * color_lin_bg[1] + 0.0722 * color_lin_bg[0]
		#
		# print("[1] Luminance values: ", Y_bg, Y_prim, Y_sec)
		# print("[2] Contrast values: ",  abs(Y_prim - Y_bg) / (Y_prim +  Y_bg) , abs(Y_sec - Y_bg) / (Y_sec + Y_bg))
		# contrast_prim = (Y_prim + 0.05) / (Y_bg + 0.05)
		# contrast_prim = contrast_prim if contrast_prim >= 1 else 1 / contrast_prim
		#
		# contrast_sec = (Y_sec + 0.05)/(Y_bg + 0.05)
		# contrast_sec= contrast_sec if contrast_sec >= 1 else 1 / contrast_sec
		#
		# print("[3] Contrast values: ", contrast_prim, contrast_sec)
		pass

	def will_bound(self, axis: int, size):
		next_coord = self.start_coords[axis] + self.axial_speeds[axis]
		bound_axis_high = next_coord + size > self.borders[1][axis]
		bound_axis_low = next_coord < self.borders[0][axis]
		# print(next_coord, bound_x, bound_y)
		return bound_axis_low or bound_axis_high

	# start_coords and end_coords change dynamically
	def calculate_next_position(self, new_size):
		# print('-----------------')
		end_coords = self.start_coords.copy()
		# start_point = coords.copy()
		for i, _ in enumerate(self.start_coords):
			if self.will_bound(axis=i, size=new_size):
				self.axial_speeds[i] *= -1
				# print("changed direction")
			# Start coordinates changed
			end_coords[i] = end_coords[i] + self.axial_speeds[i]
			# print(i, _)

			# end_coords[i] += 1
			# print(self.axial_speeds[i])
			# print(end_coords)
			# end_coords[i] += 1
		self.start_coords[:] = end_coords
		# print(self.start_coords)
		# return start_point

	# Size changes depend on frame number and rate
	def calculate_size(self, frame_number, fps, frequency=0.1):
		phase = self.amplitude * np.sin(frame_number * 2 * np.pi * frequency / fps)

		if phase >= 0:
			phase += 1
			# print(phase)
			res = abs(self.base_size * phase)
		else:
			phase -= 1
			# print(phase)
			res = abs(self.base_size / phase)

		return res

	def draw_rectangle(self, frame, frame_number, fps, cm_color=128, n=4):
		# Change base size
		# Change start coords
		self.new_size = self.calculate_size(frame_number, fps)
		# new_size = self.base_size
		self.calculate_next_position(self.new_size)
		x0, y0 = self.start_coords
		xn, yn = [c + self.new_size for c in self.start_coords]
		# print((x0, y0), (xn, yn))
		X = np.linspace(x0, xn, n + 1, dtype=int)
		Y = np.linspace(y0, yn, n + 1, dtype=int)

		colors = {1: self.color_primary, -1: self.color_secondary}
		idx = 1

		for i in range(n):
			for j in range(n):
				color = colors[idx]
				frame = cv2.rectangle(frame, (X[i], Y[j]), (X[i + 1], Y[j + 1]), color, -1)
				idx *= -1
			idx *= -1
		# frame = cv2.rectangle(frame, (int(x0), int(y0)), (int(xn), int(yn)), (cm_color, cm_color, cm_color, 255), -1)
		return frame

	def get_opposite_coords(self):
		end_coords = [c + self.new_size for c in self.start_coords]
		return self.coords_tuple(
			self.start_coords[0],
			end_coords[0],
			self.start_coords[1],
			end_coords[1]
		)