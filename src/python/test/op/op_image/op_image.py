import _flowunit as modelbox
import sys
import numpy as np
import threading
import time
import os
import cv2
from PIL import Image  


class SendExternThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, fu):
        threading.Thread.__init__(self)
        self.__fu = fu

    def run(self):
        images_files = []
        dir_or_file = os.listdir(self.__fu.path_config)
        for path in dir_or_file:
            ab_path = os.path.join(self.__fu.path_config, path)
            if os.path.isfile(ab_path) and ab_path.endswith(".jpg"):
                images_files.append(ab_path)


        while True:
            for image_file in images_files:
                img = cv2.imread(image_file)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                extern_data = self.__fu.create_external_data()
                buffer_list = extern_data.create_buffer_list()

                im_array = np.asarray(img_rgb[:,:])
                
                buffer_list.push_back(im_array)
                extern_data.send(buffer_list)

                time.sleep(0.2)
                break
            break
        self.__fu = None


class ImageFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.path_config = config.get_string("path")
        if not os.path.isdir(self.path_config):
            return modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT)

        batch = config.get_int("batch_size", 1)
        if batch != 10:
            return modelbox.Status(modelbox.Status.StatusCode.STATUS_FAULT, "invalid batch")

        self.__thread1 = SendExternThread(self)
        self.__thread1.start()

        return modelbox.Status()

    def process(self, data_ctx):
        extern_bl = data_ctx.external()
        out_bl = data_ctx.output("image_out/out_1")

        for buffer in extern_bl:
            out_bl.push_back(buffer)

        return modelbox.Status()

    def close(self):
        self.__thread1.join()
        return modelbox.Status()

    def data_pre(self, data_ctx):
        return modelbox.Status()

    def data_post(self, data_ctx):
        return modelbox.Status()

    def data_group_pre(self, data_ctx):
        return modelbox.Status()

    def data_group_post(self, data_ctx):
        return modelbox.Status()

