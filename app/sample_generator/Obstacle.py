import cv2
from collections import namedtuple

class Obstacle:
	def __init__(self, frame_width, frame_height, size, dx=0, dy=0, alpha_enabled=True):
		self.frame_width, self.frame_height = frame_width, frame_height

		# Rectangle in center when initialized
		self.base_size = size
		self.start_coords = [
			self.frame_width // 2 - self.base_size // 2 - dx,
			self.frame_height // 2 - self.base_size // 2 - dy
		]
		if alpha_enabled:
			self.img = cv2.imread("./assets/tree_branch.png", cv2.IMREAD_UNCHANGED)
		else:
			self.img = cv2.imread("./assets/brick_wall.jpg", cv2.IMREAD_UNCHANGED)
		# cv2.imshow("img", self.img)
		# print(self.img[:, :, 3])
		self.img = cv2.resize(self.img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

		self.end_coords = [
			self.start_coords[0] + self.img.shape[0],
			self.start_coords[1] + self.img.shape[1]
		]

		self.coords_tuple = namedtuple('CoordsTuple', 'x1 x2 y1 y2')
		pass

	def draw_obst(self, frame):
		# Place image with alpha-channel
		if self.img.shape[2] == 4:
			alpha_s = self.img[:, :, 3] / 255.0
			alpha_l = 1.0 - alpha_s

			for c in range(0, 3):
				# fra[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * l_img[y1:y2, x1:x2, c])
				frame[
					self.start_coords[1]:self.end_coords[1],
					self.start_coords[0]:self.end_coords[0], c]\
						= (alpha_s * self.img[:, :, c] + alpha_l * frame[
															   self.start_coords[1]:self.end_coords[1],
															   self.start_coords[0]:self.end_coords[0], c])
			return frame
		elif self.img.shape[2] == 3:
			frame[self.start_coords[1]:self.end_coords[1], self.start_coords[0]:self.end_coords[0], ] = self.img
			return frame
		else:
			raise ValueError("Unsupported image type")

	def get_opposite_coords(self):

		return self.coords_tuple(
			self.start_coords[0],
			self.end_coords[0],
			self.start_coords[1],
			self.end_coords[1]
		)