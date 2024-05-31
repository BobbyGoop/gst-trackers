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
from app.SiamMask.utils.config_helper import proccess_loss, add_default
from app.SiamMask.utils.load_helper import load_pretrain
# from app.SiamMask.experiments.siammask_sharp.custom import Custom
from app.SiamMask.tools.test import siamese_init, siamese_track
from app.gst_utils.SelectSquare import SelectSquare
from app.gst_utils.gst_hacks import map_gst_buffer, get_buffer_size

import numpy as np
import torch

import os
import json
import timeit

Gst.init(None)


class GstSiamFilter(GstBase.BaseTransform):
    name = 'gstsiamfilter'
    description = "gst.Element Tracks object using SiamMask Neural Net"
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
                                           Gst.Caps.from_string('video/x-raw,format=RGB')))

    CHANNELS = 3  # RGB
    IS_INITIALIZED = False
    SQUARED_ROI = False
    DEVICE = 'cuda'
    CONFIG_PATH = os.path.abspath("./SiamMask/experiments/siammask_sharp/config_vot18.json")
    MODEL_PATH = os.path.abspath("./SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth")

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

        success, (width, height) = get_buffer_size(self.srcpad.get_current_caps())
        if not success:
            # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.FlowReturn
            return Gst.FlowReturn.ERROR

        with map_gst_buffer(inbuffer, Gst.MapFlags.READ) as mapped:
            inp_frame = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)

        # YOUR IMAGE PROCESSING FUNCTION
        # BEGIN
        out_frame = inp_frame.copy()
        if not self.IS_INITIALIZED:
            cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
            try:
                init_rect = cv2.selectROI('SiamMask', cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), False, False)
                x, y, w, h = init_rect

            except:
                exit()
            start_time = timeit.default_timer()
            self.torch_device = torch.device(self.DEVICE)
            torch.backends.cudnn.benchmark = True

            # config = namedtuple('config')
            # Setup Model
            cfg = self.load_config({
                "config": self.CONFIG_PATH
            })

            from experiments.siammask_sharp.custom import Custom

            siammask = Custom(anchors=cfg['anchors'])
            if self.MODEL_PATH:
                assert os.path.isfile(self.MODEL_PATH), 'Please download {} first.'.format(self.MODEL_PATH)
                siammask = load_pretrain(siammask, self.MODEL_PATH)

            siammask.eval().to(self.torch_device)
            # Select ROI

            # init tracker
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            # out_frame = inp_frame.copy()
            self.state = siamese_init(out_frame, target_pos, target_sz, siammask, cfg['hp'], device=self.torch_device)
            self.IS_INITIALIZED = True
            cv2.destroyWindow('SiamMask')
            pass
        else:
            start_time = timeit.default_timer()
            # track
            self.state = siamese_track(self.state, out_frame, mask_enable=True, refine_enable=True, device=self.torch_device)
            location = self.state['ploygon'].flatten()
            mask = self.state['mask'] > self.state['p'].seg_thr

            out_frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * out_frame[:, :, 2]
            out_frame = cv2.polylines(out_frame, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            # cv2.imshow('SiamMask', im)
            # key = cv2.waitKey(1)
            # if key > 0:
            #     break
            pass

        end_time = timeit.default_timer()
        frame_time = end_time - start_time
        FPS = 1.0 / frame_time
        cv2.putText(out_frame, f"Frame Time: {format(frame_time, '.10f')}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out_frame, f"FPS: {int(FPS)}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        GstSiamFilter.save_results(frame_time, FPS)
        # END

        # HACK: force the query to be writable by messing with the refcount
        # https://bugzilla.gnome.org/show_bug.cgi?id=746329
        refcount = outbuffer.mini_object.refcount
        outbuffer.mini_object.refcount = 1

        with map_gst_buffer(outbuffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
            out = np.ndarray((height, width, self.CHANNELS), buffer=mapped, dtype=np.uint8)
            # Assign processed IN np.array to OUT np.array
            out[:] = out_frame

        # HACK: Sync timestamps and duration to preserve framerate
        outbuffer.pts = inbuffer.pts
        outbuffer.duration = inbuffer.duration
        # HACK: decrement refcount value
        outbuffer.mini_object.refcount += refcount - 1
        return Gst.FlowReturn.OK

    def load_config(self, args):
        assert os.path.exists(args['config']), '"{}" not exists'.format(args['config'])
        config = json.load(open(args['config']))

        # deal with network
        if 'network' not in config:
            print('Warning: network lost in config. This will be error in next version')

            config['network'] = {}

            if not args['arch']:
                raise Exception('no arch provided')
        args['arch'] = config['network']['arch']

        # deal with loss
        if 'loss' not in config:
            config['loss'] = {}

        proccess_loss(config['loss'])

        # deal with lr
        if 'lr' not in config:
            config['lr'] = {}
        default = {'feature_lr_mult': 1.0, 'rpn_lr_mult': 1.0, 'mask_lr_mult': 1.0, 'type': 'log', 'start_lr': 0.03}
        default.update(config['lr'])
        config['lr'] = default

        # clip
        if 'clip' in config or 'clip' in args.keys():
            if 'clip' not in config:
                config['clip'] = {}
            config['clip'] = add_default(config['clip'], {'feature': args['clip'], 'rpn': args['clip'], 'split': False})
            if config['clip']['feature'] != config['clip']['rpn']:
                config['clip']['split'] = True
            if not config['clip']['split']:
                args['clip'] = config['clip']['feature']

        return config

    @staticmethod
    def save_results(ft, fr):
        return ft, fr

    @classmethod
    def self_register(cls, plugin):
        # https://lazka.github.io/pgi-docs/#GObject-2.0/functions.html#GObject.type_register
        type_to_register = GObject.type_register(cls)

        # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Element.html#Gst.Element.register
        return Gst.Element.register(plugin, cls.name, 0, type_to_register)

