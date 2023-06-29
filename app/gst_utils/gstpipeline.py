import logging

import gi
from gi.repository import Gst, GLib
gi.require_version('Gst', '1.0')


class GstPipeline(object):
    """
        Base class to initialize any Gstreamer Pipeline from string
    """
    loop: GLib.MainLoop

    def __init__(self, command, loop):
        """
            :param command: gstreamer plugins string
            :type command: str

            :param loop: event loop to send bus messages
            :type loop: GLib.MainLoop
        """

        if not isinstance(command, str):
            raise ValueError("Invalid type. {} != {}".format(type(command), 
                                                             "str"))
        
        super(GstPipeline, self).__init__()

        self._pipeline = None
        self._active = False

        logging.info('%s %s', 'gst-launch-1.0', command)

        """
            Gsteamer Pipeline
            https://gstreamer.freedesktop.org/documentation/application-development/introduction/basics.html
        """
        self._pipeline: Gst.Pipeline
        self._pipeline = Gst.parse_launch(command)

        if not isinstance(self._pipeline, Gst.Pipeline):
            raise ValueError("Invalid type. {} != {}".format(type(self._pipeline), 
                                                             "Gst.Pipeline"))

        """
            Gsteamer Message Bus
            https://gstreamer.freedesktop.org/documentation/application-development/basics/bus.html
        """
        self._bus = self._pipeline.get_bus()  
        self._bus.add_signal_watch()
        self._bus.connect("message", self._bus_call, loop)
    
    @staticmethod
    def create_element(name):
        """
            Creates Gstreamer element

            :param name: https://gstreamer.freedesktop.org/documentation/plugins.html
            :type name: str

            :rtype: Gst.Element
        """  
        return Gst.ElementFactory.make(name)

    def get_element(self, name):
        """
            Get Gst.Element from pipeline by name

            :param name:
            :type name: str

            :rtype: Gst.Element
        """  
        element = self._pipeline.get_by_name(name)
        return element is not None, element 

    def start(self):        
        # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.StateChangeReturn
        self._pipeline.set_state(Gst.State.PLAYING)
        # self.loop.quit()

    def stop(self):
        # https://lazka.github.io/pgi-docs/Gst-1.0/enums.html#Gst.StateChangeReturn
        # self._pipeline.send_event(Gst.Event.new_eos())
        self._pipeline.set_state(Gst.State.NULL)

    def bus(self):
        return self._bus    
    
    def pipeline(self):
        return self._pipeline
    
    def _bus_call(self, bus, message, loop):
        mtype = message.type

        """
            Gstreamer Message Types and how to parse
            https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
        """
        if mtype == Gst.MessageType.EOS:
            self.stop()
            loop.quit()
            
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logging.error("{0}: {1}".format(err, debug))      
            self.stop()
            loop.quit()

        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logging.warning("{0}: {1}".format(err, debug))

        elif mtype == Gst.MessageType.INFO:
            err, debug = message.parse_warning()
            logging.warning("{0}: {1}".format(err, debug))
        return True   

