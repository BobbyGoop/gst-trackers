''' MOSSE tracking sample  This sample implements correlation-based tracking approach, described in [1].  Usage:   mosse.py [--pause] [<video source>]
 
   --pause  -  Start with playback paused at the first video frame.
               Useful for tracking target selection.
 
   Draw rectangles around objects with a mouse to track them.
 
 Keys:
   SPACE    - pause video
   c        - clear targets
 
 [1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
     http://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
 '''

# Python 2/3 compatibility
from __future__ import print_function

import sys

PY3 = sys.version_info[0] == 3

import numpy as np
import cv2 as cv


def rnd_warp(a):
	h, w = a.shape[:2]
	T = np.zeros((2, 3))
	coef = 0.2
	ang = (np.random.rand() - 0.5) * coef
	c, s = np.cos(ang), np.sin(ang)
	T[:2, :2] = [[c, -s], [s, c]]
	T[:2, :2] += (np.random.rand(2, 2) - 0.5) * coef
	c = (w / 2, h / 2)
	T[:, 2] = c - np.dot(T[:2, :2], c)
	return cv.warpAffine(a, T, (w, h), borderMode=cv.BORDER_REFLECT)


def divSpec(A, B):
	Ar, Ai = A[..., 0], A[..., 1]
	Br, Bi = B[..., 0], B[..., 1]
	C = (Ar + 1j * Ai) / (Br + 1j * Bi)
	C = np.dstack([np.real(C), np.imag(C)]).copy()
	return C


eps = 1e-5


class MOSSE:
	psr_success_level = 8

	def __init__(self, frame, rect, psr_level):
		x1, y1, w, h = rect
		x2, y2 = x1 + w, y1 + h
		# w, h = map(cv.getOptimalDFTSize, [x2 - x1, y2 - y1])
		print(w, h)
		x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
		self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
		self.size = w, h
		img = cv.getRectSubPix(frame, (w, h), (x, y))

		self.win = cv.createHanningWindow((w, h), cv.CV_32F)
		g = np.zeros((h, w), np.float32)
		g[h // 2, w // 2] = 1
		g = cv.GaussianBlur(g, (-1, -1), 2.0)
		g /= g.max()

		self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
		self.H1 = np.zeros_like(self.G)
		self.H2 = np.zeros_like(self.G)
		for _i in range(128):
			a = self.preprocess(rnd_warp(img))
			A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
			self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)
			self.H2 += cv.mulSpectrums(A, A, 0, conjB=True)
		self.update_kernel()
		self.update(frame)

	def update(self, frame, rate=0.125):
		(x, y), (w, h) = self.pos, self.size
		self.last_img = img = cv.getRectSubPix(frame, (w, h), (x, y))
		img = self.preprocess(img)
		self.last_resp, (dx, dy), self.psr = self.correlate(img)
		self.good = self.psr > self.psr_success_level
		if not self.good:
			return

		self.pos = x + dx, y + dy
		self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
		img = self.preprocess(img)

		A = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
		H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
		H2 = cv.mulSpectrums(A, A, 0, conjB=True)
		self.H1 = self.H1 * (1.0 - rate) + H1 * rate
		self.H2 = self.H2 * (1.0 - rate) + H2 * rate
		self.update_kernel()

	@property
	def state_vis(self):
		f = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
		h, w = f.shape
		f = np.roll(f, -h // 2, 0)
		f = np.roll(f, -w // 2, 1)
		kernel = np.uint8((f - f.min()) / f.ptp() * 255)
		resp = self.last_resp
		resp = np.uint8(np.clip(resp / resp.max(), 0, 1) * 255)
		vis = np.hstack([self.last_img, kernel, resp])
		return vis

	def draw_state(self, vis):
		(x, y), (w, h) = self.pos, self.size
		x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
		cv.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)
		if self.good:
			cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
		else:
			cv.line(vis, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)
			cv.line(vis, (x2, y1), (x1, y2), (0, 255, 0), thickness=4)
		x, y = (x1, y2 + 16)
		cv.putText(vis, 'PSR: %.2f' % self.psr, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
		cv.putText(vis, 'PSR: %.2f' % self.psr, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

	def preprocess(self, img):
		img = np.log(np.float32(img) + 1.0)
		img = (img - img.mean()) / (img.std() + eps)
		return img * self.win

	def correlate(self, img):
		C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
		resp = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
		h, w = resp.shape
		_, mval, _, (mx, my) = cv.minMaxLoc(resp)
		side_resp = resp.copy()
		cv.rectangle(side_resp, (mx - 10, my - 10), (mx + 10, my + 10), 0, -1)
		smean, sstd = side_resp.mean(), side_resp.std()
		psr = (mval - smean) / (sstd + eps)
		return resp, (mx - w // 2, my - h // 2), psr

	def update_kernel(self):
		self.H = divSpec(self.H1, self.H2)
		self.H[..., 1] *= -1
