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

        data_ctx.set_private("float_test", 0.5)
        data_ctx.set_private("string_test", "TEST")
        data_ctx.set_private("int_test", 100)
        data_ctx.set_private("bool_test", False)

        data_ctx.set_private("list_int_test", [1, 1, 1])
        data_ctx.set_private("list_float_test", [0.1, 0.2, 0.3])
        data_ctx.set_private("list_bool_test", [False, False, True])
        data_ctx.set_private("list_string_test", ["TEST1", "TEST2", "TEST3"])

        data_ctx.set_private("list2_int_test", [[1, 2], [3, 4]])
        data_ctx.set_private("list2_float_test", [[1.1, 2.2], [3.3, 4.4]])
        data_ctx.set_private("list2_bool_test", [[True, False], [False, True]])
        data_ctx.set_private("list2_string_test", [["hello", "world"], ["good", "bad"]])

        data_ctx.set_private("dict", {"1":1, "2":2})

        np_test = np.array([[1, 2 ,3], [11, 12, 13]])
        data_ctx.set_private("np_test", np_test)

        empty_np = np.array([])
        empty_buffer = self.create_buffer(empty_np)
        first_buffer = in_bl.front()
        last_buffer = in_bl.back()
        for buffer in in_bl:
            np_image = np.array(buffer, copy= False)
            image = Image.fromarray(np_image)

            brightness_array = np.array(image)
            add_buffer = modelbox.Buffer(self.get_bind_device(), brightness_array)
            add_buffer.copy_meta(buffer)
            add_buffer.set("float_test", data_ctx.get_private("float_test"))
            add_buffer.set("string_test", data_ctx.get_private("string_test"))
            add_buffer.set("int_test", data_ctx.get_private("int_test"))
            add_buffer.set("bool_test", data_ctx.get_private("bool_test"))

            add_buffer.set("list_int_test", data_ctx.get_private("list_int_test"))
            add_buffer.set("list_float_test", data_ctx.get_private("list_float_test"))
            add_buffer.set("list_bool_test", data_ctx.get_private("list_bool_test"))
            add_buffer.set("list_string_test", data_ctx.get_private("list_string_test"))

            add_buffer.set("list2_int_test", data_ctx.get_private("list2_int_test"))
            add_buffer.set("list2_float_test", data_ctx.get_private("list2_float_test"))
            add_buffer.set("list2_bool_test", data_ctx.get_private("list2_bool_test"))
            add_buffer.set("list2_string_test", data_ctx.get_private("list2_string_test"))

            add_buffer.set("np_test", data_ctx.get_private("np_test"))

            add_buffer.set("map_test", {"test" : 1})
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
