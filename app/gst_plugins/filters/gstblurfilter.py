# sys.path.insert(1, '../utils')

import numpy as np
import cv2
import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

GST_BLUR_FILTER = 'gstblurfilter'


def _write(message, filename, mode):
    with open(filename, mode) as handle:
        handle.write(message)


# https://lazka.github.io/pgi-docs/GstBase-1.0/classes/BaseTransform.html
class GstBlurFilter(GstBase.BaseTransform):
    name = 'gstblurfilter'
    description = "gst.Element Blurs image with Gaussian blur"
    version = '1.0.0'
    gst_license = 'LGPL'
    source_module = 'gstreamer'
    package = name
    origin = 'lifestyletransfer.com'



    __gstmetadata__ = ("An example plugin of GstBlurFilter",
                       "gst-filter/gstblurfilter.py",
                       "gst.Element blurs frame",
                       "Taras at LifeStyleTransfer.com")

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                            Gst.PadDirection.SRC,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]")),
                        Gst.PadTemplate.new("sink",
                                            Gst.PadDirection.SINK,
                                            Gst.PadPresence.ALWAYS,
                                            Gst.Caps.from_string("video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]")))

    CHANNELS = 3  # RGB

    def __init__(self):
        super(GstBlurFilter, self).__init__()  

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True

    def do_transform(self, inbuffer, outbuffer):
        """
            Implementation of simple filter.
            Inbuffer, outbuffer are different buffers, so 
            manipulations with inbuffer not affects outbuffer

            Read more:
            https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-libs/html/GstBaseTransform.html
        """

        # success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
        # if not success:
        #     # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
        #     return Gst.FlowReturn.ERROR


        # print(outbuffer.pts)
        with inbuffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:


            frame = np.ndarray((self.height, self.width, self.CHANNELS), buffer=mapped.data, dtype=np.uint8)
        # print("PLUGIN BUFFER IN: ", sys.getsizeof(mapped.data), inbuffer.pts, inbuffer.duration)
        # YOUR IMAGE PROCESSING FUNCTION
        # BEGIN
        kernel = 25
        sigma = 10
        frame = cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
        # frame = gaussian_blur(frame, 25, sigma=(10, 10))

        # END

        # HACK: force the query to be writable by messing with the refcount
        # https://bugzilla.gnome.org/show_bug.cgi?id=746329
        refcount = outbuffer.mini_object.refcount
        outbuffer.mini_object.refcount = 1

        with outbuffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
            out = np.ndarray((self.height, self.width, self.CHANNELS), buffer=mapped.data, dtype=np.uint8)
            # Assign processed IN np.array to OUT np.array
            out[:] = frame

        # HACK: decrement refcount value
        outbuffer.mini_object.refcount += refcount - 1

        # HACK: Sync timestamps and duration to preserve framerate
        outbuffer.pts = inbuffer.pts
        outbuffer.duration = inbuffer.duration

        # print("PLUGIN BUFFER OUT: ", sys.getsizeof(mapped.data), outbuffer.pts, outbuffer.duration)
        return Gst.FlowReturn.OK

    @classmethod
    def self_register(cls, plugin):
        # https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
        type_to_register = GObject.type_register(cls)

        # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
        return Gst.Element.register(plugin, cls.name, 0, type_to_register)
