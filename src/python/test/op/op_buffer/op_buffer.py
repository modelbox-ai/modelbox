import _flowunit as modelbox
import sys
import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance

class BufferTestFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.__brightness = config.get_float("buffer_config", 0.0)
        if self.__brightness != 0.2:
            return modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT)

        return modelbox.Status()

    def process(self, data_ctx):
        in_bl = data_ctx.input("buffer_in")
        out_bl = data_ctx.output("buffer_out")

        empty_np = np.array([])
        empty_buffer = self.create_buffer(empty_np)

        for buffer in in_bl:
            np_image = np.array(buffer, copy= False)
            image = Image.fromarray(np_image)

            brightness_array = np.array(image)
            add_buffer = modelbox.Buffer(self.get_bind_device(), brightness_array)
            add_buffer.copy_meta(buffer)
            
            add_buffer.set("float_test", 0.5)
            add_buffer.set("string_test", "TEST")
            add_buffer.set("int_test", 100)
            add_buffer.set("bool_test", False)
            add_buffer.set("list_int_test", [1, 1, 1])
            add_buffer.set("list_float_test", [0.1, 0.2, 0.3])
            add_buffer.set("list_bool_test", [False, False, True])
            add_buffer.set("list_string_test", ["TEST1", "TEST2", "TEST3"])
            try:
                add_buffer.set("map_test", {"test" : 1})
            except ValueError as err:
                modelbox.info(str(err))
            else:
                return modelbox.Status.StatusCode.STATUS_SHUTDOWN

            out_bl.push_back(add_buffer)

        return modelbox.Status()


    def close(self):
        return modelbox.Status()

    def data_pre(self, data_ctx):
        return modelbox.Status()

    def data_post(self, data_ctx):
        return modelbox.Status()

    def data_group_pre(self, data_ctx):
        return modelbox.Status()

    def data_group_post(self, data_ctx):
        return modelbox.Status()
