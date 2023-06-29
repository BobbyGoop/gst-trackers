import timeit
import time
import os
import cv2
import numpy as np
import typing
from fractions import Fraction
import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstAudio', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstBase, GstVideo, GLib, GstAudio

from gst_utils.SelectSquare import SelectSquare
from gst_utils.gst_hacks import map_gst_buffer, get_buffer_size
from mosse_filter.mosse_opencv import MOSSE
from sample_generator.CustomConfig import CustomConfig
from sample_generator.MozaicSquare import MosaicSquare
from sample_generator.Obstacle import Obstacle

GST_MOSSE_FILTER = 'gstmossefilter'


def _write(message, filename, mode):
	with open(filename, mode) as handle:
		handle.write(message)


OCAPS = Gst.Caps.from_string (
        'video/x-raw, format=RGB, framerate=30/1')


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_GRAYSCALE = False
SAMPLESPERBUFFER = 1024
# 60s * 30 fps = 1800 frames
DEFAULT_DURATION = 1800

class GstSampleSrc(GstBase.BaseSrc):
	name = 'gstsamplesrc'
	description = "gst.Src Example source"
	version = '1.0.0'
	gst_license = 'LGPL'
	source_module = 'gstreamer'
	package = name
	origin = 'lifestyletransfer.com'

	CHANNELS = 3

	__gstmetadata__ = ('CustomSrc', 'Src', 'Custom test src element', 'Mathieu Duponchelle')

	__gsttemplates__ = Gst.PadTemplate.new(
		"src",
		Gst.PadDirection.SRC,
		Gst.PadPresence.ALWAYS,
		Gst.Caps.from_string("video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647],framerate=[0/1,2147483647/1]")
	)


	__gproperties__ = {
		# "video-duration" : (
		# 	int,
		# 	"Duration",
		# 	"Duration of the video in seconds",
		# 	DEFAULT_DURATION,
		# 	GObject.ParamFlags.READWRITE
		# ),
		"is-grayscale": (
			bool,
			"Gray",
			"Generate Grayscale image",
			DEFAULT_GRAYSCALE,
			GObject.ParamFlags.READWRITE
		)
	}


	def __init__(self):

		self.gray = DEFAULT_GRAYSCALE

		# self.info = GstVideo.VideoInfo()
		self.set_live(False)
		self.set_format(Gst.Format.TIME)
		self.pts:  GLib.MAXUINT64
		self.pts = 0
		self.stream_duration = DEFAULT_DURATION
		# print(os.path.abspath("config.json"))
		cfg = CustomConfig.load_json(os.path.abspath("configs/config.json"))
		self._VID = cfg.video
		self._OBJ = cfg.object
		self._BAR = cfg.obstacle

		if self._BAR.enabled:
			self.obst = Obstacle(self._VID.width, self._VID.height, self._BAR.size, alpha_enabled=self._BAR.alpha_enabled)
		else:
			self.obst = None

		colors = []
		colors.append([(204, 0, 204, 255),
					   (0, 255, 128, 255)])
		colors.append([(21, 131, 255, 255),
					   (174, 229, 23, 255)])
		colors.append([(255, 0, 127, 255),
					   (0, 255, 255, 255)])
		colors.append([(255, 0, 0, 255),
					   (51, 153, 255, 255)])
		colors.append([(0, 0, 255, 255),
					   (0, 255, 0, 255)])
		colors.append([(119, 29, 170, 255),
					   (0, 250, 165, 255)])

		self.objects = []
		for i in range(self._OBJ.number):
			if i == 0:
				self.objects.append(
					MosaicSquare(
						self._VID.width,
						self._VID.height,
						self._OBJ.size,
						self._OBJ.pulsation,
						self._VID.border,
						self._OBJ.speed,
						dx=self._OBJ.center_offset,
						dy=0,
						base=True
					)
				)
			else:
				self.objects.append(
					MosaicSquare(
						self._VID.width,
						self._VID.height,
						self._OBJ.size,
						self._OBJ.pulsation,
						self._VID.border,
						speed=np.random.randint(5, 35),
						dx=np.random.randint(-self._VID.width // 3, self._VID.width // 3),
						dy=np.random.randint(-self._VID.height // 3, self._VID.height // 3),
						base=False,
						colors = colors[i-1]
					)
				)
			time.sleep(1)
		print([i.get_opposite_coords() for i in self.objects])

		self.FRAMES_NUMBER = DEFAULT_DURATION
		self.CURRENT_FRAME = 0
		super(GstSampleSrc, self).__init__()
		# self.set_format(Gst.Format.TIME)

	def do_set_caps(self, caps: Gst.Caps):
		caps_struct: Gst.Structure
		caps_struct = caps.get_structure(0)
		(success, self.width) = caps_struct.get_int('width')
		if not success:
			raise ValueError("Invalid captions param")
		(success, self.height) = caps_struct.get_int('height')
		if not success:
			raise ValueError("Invalid captions param")

		self.framerate: Fraction
		self.framerate = Fraction(str(caps_struct.get_value('framerate')))
		print('Read caps: ', self.width, self.height, self.framerate)

		if self.framerate.numerator != self._VID.framerate:
			# raise ValueError()
			print(f"Captions framerate [{self.framerate}] overrides Config framerate [{self._VID.framerate}]")

		# self.info.new_from_caps(caps)
		# self.info.from_caps(caps)
		# caps_struct: Gst.Structure
		# caps_struct = caps.get_structure(0)
		# width = caps_struct.get_value('width')
		# height = caps_struct.get_int('height')
		# frac = type(caps_struct.get_value('framerate'))

		# self.info.b
		# print(self.info.size)
		# print(self.info.__dict__)
		# self.set_blocksize(self.info.size)
		return True

	def do_get_property(self, prop: GObject.Property):
		# if prop.name == 'video-duration':
		# 	return self.stream_duration
		if prop.name == 'is-grayscale':
			return self.gray
		else:
			raise AttributeError('Unknown property %s' % prop.name)

	def do_set_property(self, prop: GObject.Property, value: typing.Any):
		# if prop.name == 'video-duration':
		# 	self.stream_duration = value
		if prop.name == 'is-grayscale':
			self.gray = value
		else:
			raise AttributeError('Unknown property %s' % prop.name)

	# def do_start(self):
	# 	self.next_sample = 0
	# 	self.next_byte = 0
	# 	self.next_time = 0
	# 	self.accumulator = 0
	# 	self.generate_samples_per_buffer = SAMPLESPERBUFFER
	#
	# 	return True

	# def do_gst_base_src_query(self, query):
	# 	if query.type == Gst.QueryType.LATENCY:
	# 		latency = Gst.util_uint64_scale_int(self.generate_samples_per_buffer, Gst.SECOND, self.info.rate)
	# 		is_live = self.is_live
	# 		query.set_latency(is_live, latency, Gst.CLOCK_TIME_NONE)
	# 		res = True
	# 	else:
	# 		res = GstBase.BaseSrc.do_query(self, query)
	# 	return res

	# def do_get_times(self, buf):
	# 	end = 0
	# 	start = 0
	# 	if self.is_live:
	# 		ts = buf.pts
	# 		if ts != Gst.CLOCK_TIME_NONE:
	# 			duration = buf.duration
	# 			if duration != Gst.CLOCK_TIME_NONE:
	# 				end = ts + duration
	# 			start = ts
	# 	else:
	# 		start = Gst.CLOCK_TIME_NONE
	# 		end = Gst.CLOCK_TIME_NONE
	#
	# 	return start, end

	def do_fill(self, offset, length, buf: Gst.Buffer):
		# success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
		# print(success, width, height)
		# HACK: force the query to be writable by messing with the refcount
		# https://bugzilla.gnome.org/show_bug.cgi?id=746329
		# print(self.height, self.width)
		# print(offset, length, buf.get_size())

		# print(self.pts)
		# refcount = buf.mini_object.refcount
		# buf.mini_object.refcount = 1

		# Generate frames
		# DEFAUlt: Stable 30 FPS
		start_time = timeit.default_timer()
		# data = np.zeros(shape=(self.height, self.width,  self.CHANNELS)) + [255, 84, 170]
		data = np.zeros(shape=(self.height, self.width, self.CHANNELS)) + 192
		squares=4
		for sq in self.objects:
			data = sq.draw_rectangle(data, self.CURRENT_FRAME, self.framerate.numerator, 128, n=squares)
			# squares=2

			# coords_sq.append(sq.get_opposite_coords())

		if self.obst:
			data = self.obst.draw_obst(data)
			# if len(coords_sq) == 1:
			# 	res = overlap(coords_sq.pop(), obst.get_opposite_coords())

		if self._VID.noise > 0:
			data += np.random.normal(0, self._VID.noise, data.shape)

		# if self._OBJ.contrast_mode.enabled:
		# 	lab = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_BGR2LAB)
		# 	l_channel, a, b = cv2.split(lab)
		# 	clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
		# 	cl = clahe.apply(l_channel)
		# 	limg = cv2.merge((cl, a, b))
		# 	data = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

		if self._VID.grayscale:
			data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_BGR2GRAY)

		data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_BGR2RGB)
		end_time = timeit.default_timer()


		# cv2.imwrite(f'assets/{self.CURRENT_FRAME:05d}.jpg', data)
		# Measure time and calculate FPS

		target_fps = self.framerate.numerator
		# target_fps = 30

		self.CURRENT_FRAME += 1
		try:
			with buf.map(Gst.MapFlags.WRITE) as info:
				A = np.ndarray(shape=(self.height, self.width,  3), dtype=np.uint8, buffer=info.data)
				A[:] = data
				if self.CURRENT_FRAME > self.FRAMES_NUMBER:
					return Gst.FlowReturn.EOS
			# buf.mini_object.refcount += refcount - 1
		except Gst.MapError as e:
			Gst.error("Mapping error: %s" % e)
			return Gst.FlowReturn.ERROR



		real_fps = int(1.0 / (end_time - start_time))
		# print(real_fps, target_fps)
		# print(target_fps, FPS)
		# if FPS >= 60:
		# 	target_fps = 60
		if 10 <= real_fps < target_fps:
			target_fps = 10
		elif real_fps < 10:
			target_fps = 5
		# print(FPS)
		# if real_fps < target_fps:
		# 	target_fps = real_fps
		buf_duration = 10 ** 9 / target_fps
		# print(buf_duration)
		# buf_duration = 10 ** 9 / FPS
		self.pts += buf_duration


		buf.pts = self.pts
		buf.duration = buf_duration
		# print("SRC BUFFER: ", sys.getsizeof(info.data), buf.duration)
		return Gst.FlowReturn.OK, buf
		# if length == -1:
		# 	samples = SAMPLESPERBUFFER
		# else:
		# 	samples = int(length / self.info.bpf)

		# samples = SAMPLESPERBUFFER
		# self.generate_samples_per_buffer = samples
		#
		# # bytes_ = samples * self.info.bpf
		# bytes_ = SAMPLESPERBUFFER
		# next_sample = self.next_sample + samples
		# next_byte = self.next_byte + bytes_
		# next_time = Gst.util_uint64_scale_int(next_sample, Gst.SECOND, self.info.)

		# if not self.mute:
		# 	r = np.repeat(np.arange(self.accumulator, self.accumulator + samples), self.info.channels)
		# 	data = ((np.sin(2 * np.pi * r * self.freq / self.info.rate) * self.volume).astype(np.float32))
		# else:
		# 	data = [0] * bytes_
		# data = np.zeros((self.height, self.width, 3)) + 128
		# data += np.random.normal(0, 2, (self.height, self.width, 3))
		#
		# buffer = Gst.Buffer.new_wrapped(bytes(data))
		#
		# buffer.offset = self.next_sample
		# buffer.offset_end = next_sample
		# buffer.pts = self.next_time
		# buffer.duration = next_time - self.next_time
		#
		# self.next_time = next_time
		# self.next_sample = next_sample
		# self.next_byte = next_byte
		# self.accumulator += samples
		# self.accumulator %= self.info.rate / self.freq

		# return (Gst.FlowReturn.OK, buffer)

	@classmethod
	def self_register(cls, plugin):
		# https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
		type_to_register = GObject.type_register(cls)

		# https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
		return Gst.Element.register(plugin, cls.name, 0, type_to_register)