import _flowunit as modelbox
import sys
import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance

class BrightnessFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()


    def open(self, config):
        self.__brightness = config.get_float("brightness", 0.0)
        if self.__brightness == 0.0:
            return modelbox.Status.StatusCode.STATUS_FAULT

        return modelbox.Status()

    def process(self, data_ctx):
        in_bl = data_ctx.input("brightness_in")
        out_bl = data_ctx.output("brightness_out")

        for buffer in in_bl:
            np_image = np.array(buffer, copy= False)
            image = Image.fromarray(np_image)

            brightness_image = ImageEnhance.Brightness(image).enhance(self.__brightness)

            brightness_array = np.array(brightness_image)
            add_buffer = self.create_buffer(brightness_array)
            add_buffer.copy_meta(buffer)
            
            add_buffer.set("brightness", self.__brightness)

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
