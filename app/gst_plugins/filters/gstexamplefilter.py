#!/usr/bin/python3
# exampleTransform.py
# 2019 Daniel Klamt <graphics@pengutronix.de>

# Inverts a grayscale frame in place, requires numpy.
#
# gst-launch-1.0 videotestsrc ! ExampleTransform ! videoconvert ! xvimagesink
import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GObject, GstBase, GstVideo

import numpy as np

Gst.init(None)


class GstExampleFilter(GstBase.BaseTransform):
    name = 'gstexamplefilter'
    description = "gst.Element Example Image Transform"
    version = '1.0.0'
    gst_license = 'LGPL'
    source_module = 'gstreamer'
    package = name
    origin = 'lifestyletransfer.com'

    __gstmetadata__ = ('ExampleTransform Python', 'Transform',
                      'example gst-python element that can modify the buffer gst-launch-1.0 videotestsrc ! ExampleTransform ! videoconvert ! xvimagesink', 'dkl')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           Gst.Caps.from_string('video/x-raw,format=RGB')),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           Gst.Caps.from_string('video/x-raw,format=RGB'))
                        )

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True

    def do_transform_ip(self, buf: Gst.Buffer):
        try:
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                # Create a NumPy ndarray from the memoryview and modify it in place:
                A = np.ndarray(shape = (self.height, self.width, 3), dtype = np.uint8, buffer = info.data)
                kernel = 25
                sigma = 10
                A[:] = cv2.GaussianBlur(A, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)

            return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR

    @classmethod
    def self_register(cls, plugin):
        # https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
        type_to_register = GObject.type_register(cls)

        # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
        return Gst.Element.register(plugin, cls.name, 0, type_to_register)
