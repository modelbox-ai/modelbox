#
# Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import http.client
import json
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent

class HttpConfig:
    def __init__(self, img_base64_str):
        self.hostIP = "127.0.0.1"
        self.Port = 8190

        self.httpMethod = "POST"
        self.requstURL = "/v1/mnist_test"

        self.headerdata = {
            "Content-Type": "application/json"
        }

        self.test_data = {
            "image_base64": img_base64_str
        }

        self.body = json.dumps(self.test_data)

img_path = str(source_dir) + "/mnist_0.png"

with open(img_path, 'rb') as fp:
    base64_data = base64.b64encode(fp.read())
    img_base64_str = str(base64_data, encoding='utf8')

http_config = HttpConfig(img_base64_str)

conn = http.client.HTTPConnection(host=http_config.hostIP, port=http_config.Port)
conn.request(method=http_config.httpMethod, url=http_config.requstURL, body=http_config.body,
             headers=http_config.headerdata)

response = conn.getresponse().read().decode()
print(response)
