import _flowunit as modelbox
import sys
import numpy as np
import os
import cv2
from PIL import Image
from PIL import ImageChops

class ShowFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.__out_path_config = config.get_string("out_path")
        if not os.path.exists(self.__out_path_config):
            os.mkdir(self.__out_path_config)

        self.__is_save_config = config.get_bool("is_save")
        if self.__is_save_config == True:
            self.__out_path_config = config.get_string("out_path", "./")
            if not os.path.exists(self.__out_path_config):
                os.mkdir(self.__out_path_config)

            self.__out_file = self.__out_path_config + '/python_test_show_out.png'
            if os.path.exists(self.__out_file):
                os.remove(self.__out_file)

        self.__check_path = config.get_string("check_path")
        if not os.path.exists(self.__check_path):
            return modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT, "invalid check file path, it is not exist")

        return modelbox.Status()

    def process(self, data_ctx):
        in_bl = data_ctx.input("show_in")

        for buffer in in_bl:
            np_image = np.array(buffer, copy= False)

            if np_image.shape[0] != 360:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid height")

            if np_image.shape[1] != 480:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid width")

            if np_image.shape[2] != 3:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid channels")

            brightness = buffer.get("brightness")
            if brightness != 0.1:
                return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid brightness")

            image = Image.fromarray(np_image)

            # if self.__is_save_config == True:
            #    image.save(self.__out_file)

            with Image.open(self.__check_path) as check_image:
                try:
                    diff = ImageChops.difference(image, check_image)
                    if diff.getbbox() is not None:
                        return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid check image")
                except ValueError as e:
                    return modelbox.Status(modelbox.Status.StatusCode.STATUS_SHUTDOWN, "invalid check image size")

            # check_image.close()

        modelbox.info("ShowFlowunit process")
        return modelbox.Status(modelbox.Status.StatusCode.STATUS_STOP)

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

