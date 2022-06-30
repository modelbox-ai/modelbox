import _flowunit as modelbox
import sys
import os

class ArgsFlowunit(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.test_arg1 = config.get_string("test_arg1")
        self.test_arg2 = config.get_string("test_arg2")

        modelbox.info("get test_arg1", self.test_arg1)
        modelbox.info("get test_arg2", self.test_arg2)

        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_ctx):
        inputs = data_ctx.input("input")
        outputs = data_ctx.output("output")

        for buffer in inputs:
            input_str = buffer.as_object()
            modelbox.info("get input", input_str)
            result = input_str + ", " + self.test_arg1 + ", " + self.test_arg2
            modelbox.info("generate result", result)
            out_buffer = modelbox.Buffer(self.get_bind_device(), result)
            outputs.push_back(out_buffer)

        modelbox.info("ArgsFlowunit process")
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_ctx):
        return modelbox.Status()

    def data_post(self, data_ctx):
        return modelbox.Status()

