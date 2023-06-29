import argparse
import cv2

from mosse_filter.mosse_opencv import MOSSE
from mosse_filter.mosse_v2 import MosseFilterV2


class App:
	def __init__(self, video_src, paused=False):
		self.cap = cv2.VideoCapture(video_src)
		_, self.frame = self.cap.read()
		cv2.imshow('frame', self.frame)
		self.rect_sel = RectSelector('frame', self.onrect)
		self.trackers = []
		self.paused = paused

	def onrect(self, rect):
		frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		tracker = MOSSE(frame_gray, rect)
		self.trackers.append(tracker)

	def run(self):
		while True:
			if not self.paused:
				ret, self.frame = self.cap.read()
				if not ret:
					break
				frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
				for tracker in self.trackers:
					tracker.update(frame_gray)

			vis = self.frame.copy()
			for tracker in self.trackers:
				tracker.draw_state(vis)
			if len(self.trackers) > 0:
				cv2.imshow('tracker state', self.trackers[-1].state_vis)
			self.rect_sel.draw(vis)

			cv2.imshow('frame', vis)
			ch = cv2.waitKey(10)
			if ch == 27:
				break
			if ch == ord(' '):
				self.paused = not self.paused
			if ch == ord('c'):
				self.trackers = []


if __name__ == '__main__':
	parse = argparse.ArgumentParser()
	parse.add_argument('--lr', type=float, default=0.125, help='the learning rate')
	parse.add_argument('--sigma', type=float, default=100, help='the sigma')
	parse.add_argument('--num_pretrain', type=int, default=128, help='the number of pretrain')
	parse.add_argument('--rotate', action='store_true', help='if rotate frame during pre-training.')
	parse.add_argument('--record', action='store_true', help='record the frames')
	args = parse.parse_args()
	img_path = './video/IMG_6248.MOV'

	tracker = MosseFilterV2(args, img_path)
	tracker.start_tracking()

	"""OPENCV PART"""
	# if __name__ == '__main__':
	# 	print(__doc__)
	# 	import sys, getopt
	#
	# 	opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])
	# 	opts = dict(opts)
	# 	try:
	# 		video_src = args[0]
	# 	except:
	# 		video_src = '0'
	#
	# 	App(video_src, paused='--pause' in opts).run()
