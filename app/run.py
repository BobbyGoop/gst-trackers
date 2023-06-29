import argparse
import logging
import os
import signal
# Required imports
# Gst, GstBase, GObject
import gi
import numpy as np
import pandas as pd

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

# Init Gobject Threads
# Init Gstreamer

GObject.threads_init()
Gst.init(None)
Gst.debug_set_active(True)
Gst.debug_set_default_threshold(3)

from gst_utils.gstpipeline import GstPipeline
from gst_plugins.sources.gstsamplesrc import GstSampleSrc
from gst_plugins.filters.gstsiamfilter import GstSiamFilter
from gst_plugins.filters.gstmossefilter import GstMosseFilter
from gst_plugins.filters.gstexamplefilter import GstExampleFilter
from gst_plugins.filters.gstblurfilter import GstBlurFilter


# Set logging level=DEBUG
logging.basicConfig(level=0)
FRAMES_MOSSE = np.empty([0, 3])
FRAMES_SIAM = np.empty([0, 2])

def frames_collector_mosse(func):
	def wrapper(*args, **kwargs):
		ft, fr, psr = func(*args, **kwargs)
		global FRAMES_MOSSE
		FRAMES_MOSSE = np.append(FRAMES_MOSSE, [[ft, fr, psr]], axis=0)

	return wrapper

def frames_collector_siam(func):
	def wrapper(*args, **kwargs):
		ft, fr = func(*args, **kwargs)
		print(*args)
		global FRAMES_SIAM
		FRAMES_SIAM = np.append(FRAMES_SIAM, [[ft, fr]], axis=0)

	return wrapper


def sigint_handler(sig, frame):
	if sig == signal.SIGINT:
		loop.quit()
		print(sig)
	else:
		raise ValueError("Undefined handler for '{}'".format(sig))


def register_by_name(filter):
	# Parameters explanation
	# https://lazka.github.io/pgi-docs/Gst-1.0/classes/Plugin.html#Gst.Plugin.register_static
	name = filter.name
	description = filter.description
	version = filter.version
	gst_license = filter.gst_license
	source_module = filter.source_module
	package = filter.package
	origin = filter.origin
	if not Gst.Plugin.register_static(Gst.VERSION_MAJOR, Gst.VERSION_MINOR, name, description, filter.self_register, version, gst_license, source_module, package, origin):
		raise ImportError("Plugin {} not registered".format(filter.name))
	return True


if __name__ == "__main__":
	os.environ["GST_PLUGIN_PATH"] = "C:/Users/Ivan/anaconda3/envs/gst-trackers/Library/lib/gstreamer-1.0/"
	os.environ["GST_PLUGIN_SYSTEM_PATH"] = "C:/Users/Ivan/anaconda3/envs/gst-trackers/Library/lib/gstreamer-1.0/"
	# How to use argparse:
	# https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--filter", choices=['mosse', 'siam', 'blur', 'example'], required=False, help="Path to video file")
	ap.add_argument("-r", "--rate", default=30, required=False, help="Video framerate",type=int)
	ap.add_argument("-sr", "--size_roi", default=120, required=False, help="Regulates ROI size", type=int)
	# ap.add_argument("-b", "--blur", action='store_true', help="ON/OFF blur filter")
	args = vars(ap.parse_args())

	# folder_name = args['data_folder']
	# file_name = os.path.abspath(args['file'])
	# if not os.path.isfile(file_name):
	# 	raise ValueError('File {} not exists'.format(file_name))

	# use_blur_filter = args['blur']

	# Build pipeline
	# filesrc https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-filesrc.html
	# decodebin https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-base-plugins/html/gst-plugins-base-plugins-decodebin.html
	# videoconvert https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-base-plugins/html/gst-plugins-base-plugins-videoconvert.html
	# gtksink https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-good/html/gst-plugins-good-plugins-gtksink.html
	# command = 'filesrc location={} ! '.format(file_name)
	# command = f'multifilesrc location="{folder_name}/%05d.jpg" index=0 caps="image/jpeg,framerate=\(fraction\)60/1" ! '
	framerate = args['rate']
	command = f'gstsamplesrc is-grayscale=false ! video/x-raw,width=1280,height=720,framerate={framerate}/1 ! '
	# command += 'decodebin ! '
	# command += 'nvjpegdec !'
	# command += ''
	# command += ' videoconvert ! '
	# command += 'gstsiamfilter ! '
	# command += f'videorate ! video/x-raw,framerate={framerate}/1 ! '

	# Use videorate to pass pts and duration to filter
	# command += 'videorate ! '
	# command += 'gstblurfilter ! '
	# command += 'gstexamplefilter ! '
	if args['filter'] == 'mosse':
		command += 'videorate ! gstmossefilter ! '
	elif args['filter'] == 'siam':
		command += 'videorate ! gstsiamfilter ! '
	elif args['filter'] == 'blur':
		command += 'videorate ! gstblurfilter ! '
	elif args['filter'] == 'example':
		command += 'videorate ! gstexamplefilter ! '
	else:
		pass
	# command += 'nvh264enc ! '
	# command += 'h264parse ! '
	# command += 'mp4mux ! filesink location=vid2.mp4'
	# command += 'avdec_h264 ! '
	# command += 'gtksink'
	# command += "glimagesink"
	# command += 'videorate ! video/x-raw,framerate=30/1 ! '
	command += 'videoconvert ! video/x-raw,format=RGBA ! '
	# command += 'videorate ! video/x-raw,framerate=30/1 ! '
	# command += 'autovideosink'
	# command += 'fpsdisplaysink video-sink=d3d11videosink text-overlay=true'
	command += "d3d11videosink sync=true"
	# command += "avdec_h264 ! h264parse ! mp4mux ! filesink location=vid2.mp4"
	GstMosseFilter.save_results = frames_collector_mosse(GstMosseFilter.save_results)
	GstSiamFilter.save_results = frames_collector_siam(GstSiamFilter.save_results)

	GstMosseFilter.ROI_SQUARED = True
	GstMosseFilter.ROI_SIZE = args['size_roi']

	register_by_name(GstSampleSrc)
	register_by_name(GstMosseFilter)
	register_by_name(GstSiamFilter)
	register_by_name(GstBlurFilter)
	register_by_name(GstExampleFilter)


	# command += 'x264enc ! mp4mux ! filesink location=gopro2.mp4'

	signal.signal(signal.SIGINT, sigint_handler)
	# Init GObject loop to handle Gstreamer Bus Events
	loop = GLib.MainLoop()
	pipeline = GstPipeline(command, loop)

	pipeline.start()
	loop.run()
	# print('stop')
	# pipeline.stop()

	# print("Writing frame data...")
	# Write frames info to file
	if args['filter'] == 'mosse':
		pd.DataFrame(FRAMES_MOSSE).to_csv("fps_mosse.csv", header=["frame_time", "frame_rate", "PSR"], index=False)
	elif args['filter'] == 'siam':
		pd.DataFrame(FRAMES_SIAM).to_csv("fps_siam.csv", header=["frame_time", "frame_rate"], index=False)