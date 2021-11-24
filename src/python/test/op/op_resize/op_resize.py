import _flowunit as modelbox
import sys
import numpy as np
import os
import cv2
from PIL import Image  

class ResizeFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.height_config = config.get_int("height")
        self.width_config = config.get_int("width")

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_ctx):
        in_bl = data_ctx.input("resize_in")
        out_bl = data_ctx.output("resize_out")

        for buffer in in_bl:
            np_image = np.array(buffer, copy= False)
            resize_image = Image.fromarray(np_image).resize((self.width_config, self.height_config))
            out_bl.push_back(np.array(resize_image))

        modelbox.info("ResizeFlowunit process")
        return modelbox.Status.StatusCode.STATUS_SUCCESS

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

