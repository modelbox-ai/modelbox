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

import http.client
import json

class HttpConfig:
    def __init__(self, msg):
        self.hostIP = "127.0.0.1"
        self.Port = 7770

        self.httpMethod = "POST"
        self.requstURL = "/v1/hello_world"

        self.headerdata = {
            "Content-Type": "application/json"
        }

        self.data = {
            "msg": msg
        }

        self.body = json.dumps(self.data)

        
if __name__ == "__main__":
    http_config = HttpConfig("hello world!")

    conn = http.client.HTTPConnection(host=http_config.hostIP, port=http_config.Port)
    conn.request(method=http_config.httpMethod, url=http_config.requstURL, body=http_config.body,
                headers=http_config.headerdata)

    response = conn.getresponse().read().decode()
    print(response)

