import sys
sys.path.insert(1, '../utils')
import timeit

import cv2
import gi
import numpy as np


gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

from app.gst_utils.SelectSquare import SelectSquare
from app.gst_utils.gst_hacks import map_gst_buffer, get_buffer_size
from app.mosse_filter.mosse_opencv import MOSSE

GST_MOSSE_FILTER = 'gstmossefilter'


def _write(message, filename, mode):
	with open(filename, mode) as handle:
		handle.write(message)


# https://lazka.github.io/pgi-docs/GstBase-1.0/classes/BaseTransform.html
class GstMosseFilter(GstBase.BaseTransform):
	name = 'gstmossefilter'
	description = "gst.Element Tracks object using MOSSE algorithm"
	version = '1.0.0'
	gst_license = 'LGPL'
	source_module = 'gstreamer'
	package = name
	origin = 'lifestyletransfer.com'


	__gstmetadata__ = ("Plugin for single object tracking using MOSSE algorithm", "gst-filter/gstmossefilter.py", "gst.Element blurs frame", "Taras at LifeStyleTransfer.com")

	__gsttemplates__ = (Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.from_string("video/x-raw,format=RGB")),
						Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.from_string("video/x-raw,format=RGB")))

	CHANNELS = 3  # RGB
	IS_INITIALIZED = False
	ROI_SQUARED = False
	ROI_SIZE = 100
	tracker: MOSSE

	def __init__(self):
		super(GstMosseFilter, self).__init__()

	def do_transform(self, inbuffer: Gst.Buffer, outbuffer: Gst.Buffer):
		"""
			Implementation of simple filter.
			Inbuffer, outbuffer are different buffers, so
			manipulations with inbuffer not affects outbuffer

			Read more:
			https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html
		"""

		success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
		# print("MOSSE: ", width, height)
		if not success:
			# https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
			return Gst.FlowReturn.ERROR

		with inbuffer.map(Gst.MapFlags.READ) as mapped:
			inp_frame = np.ndarray((height, width, self.CHANNELS), buffer=mapped.data, dtype=np.uint8)

		# YOUR IMAGE PROCESSING FUNCTION
		# BEGIN
		if not self.IS_INITIALIZED:
			if self.ROI_SQUARED:
				# 4k - ROI 500px
				# HD - ROI 250px
				# HD - ROI 167px
				rect_sel = SelectSquare(cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB), "Range Of Object Interest", self.ROI_SIZE).get_coords()
				# rect_sel = SelectSquare(inp_frame, "Range Of Object Interest", 167).get_coords()
			else:
				cv2.namedWindow("Range Of Object Interest", cv2.WINDOW_KEEPRATIO)
				cv2.resizeWindow("Range Of Object Interest", 1280, 720)
				rect_sel = cv2.selectROI("Range Of Object Interest", inp_frame, showCrosshair=True, fromCenter=False)
			start_time = timeit.default_timer()
			# print(rect_sel[2:])
			frame_gray = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2GRAY)
			self.tracker = MOSSE(frame_gray, rect_sel, 8)
			self.IS_INITIALIZED = True
			out_frame = inp_frame
			cv2.destroyWindow("Range Of Object Interest")
		else:
			start_time = timeit.default_timer()
			frame_gray = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2GRAY)
			self.tracker.update(frame_gray)
			vis = inp_frame.copy()
			self.tracker.draw_state(vis)

			# DRAW IN FRAME, NOT ON WINDOW!!!!
			state_win = cv2.cvtColor(self.tracker.state_vis, cv2.COLOR_GRAY2RGB)
			stats_height = 100
			aspect_ratio = stats_height/state_win.shape[0]
			state_win = cv2.resize(src=state_win, dsize=None, fx=aspect_ratio, fy=aspect_ratio, interpolation=cv2.INTER_CUBIC)

			# Place filter output to upper left corner
			vis[0:state_win.shape[0], 0:state_win.shape[1]] = state_win

			out_frame = vis  # frame = self.track_object(frame)

		end_time = timeit.default_timer()
		frame_time = end_time - start_time
		FPS = 1.0 / frame_time
		cv2.putText(out_frame, f"Frame Time: {format(frame_time, '.10f')}", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(out_frame, f"FPS: {int(FPS)}", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		self.save_results(frame_time, FPS)
		# END

		# HACK: force the query to be writable by messing with the refcount
		# https://bugzilla.gnome.org/show_bug.cgi?id=746329
		refcount = outbuffer.mini_object.refcount
		outbuffer.mini_object.refcount = 1

		with map_gst_buffer(outbuffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
			out = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)
			# Assign processed IN np.array to OUT np.array
			out[:] = out_frame

		# HACK: decrement refcount value
		outbuffer.mini_object.refcount += refcount - 1
		write_time = timeit.default_timer()

		# HACK: Sync timestamps and duration to preserve framerate
		outbuffer.pts = inbuffer.pts
		outbuffer.duration = inbuffer.duration

		# print("Write buffer time: ", 1.0/(write_time-start_time))
		return Gst.FlowReturn.OK

	def save_results(self, ft, fr):
		psr = self.tracker.psr
		return ft, fr, psr

	@classmethod
	def self_register(cls, plugin):
		# https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
		type_to_register = GObject.type_register(cls)

		# https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
		return Gst.Element.register(plugin, cls.name, 0, type_to_register)

