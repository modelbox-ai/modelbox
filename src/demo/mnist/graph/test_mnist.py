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
import sys
import os
import random
import argparse
from pathlib import Path
from PIL import Image
from urllib.parse import urlparse
import urllib.request

source_path = Path(__file__).resolve()
source_dir = source_path.parent
mnist_image_path = str(source_dir) + "/mnist-image"

def DisplayImage(image_path):
    # pass the image as command line argument
    img = Image.open(image_path)

    # resize the image
    width, height = img.size
    aspect_ratio = height/width
    new_width = 32
    new_height = aspect_ratio * new_width * 0.45
    img = img.resize((new_width, int(new_height)))

    # convert image to greyscale format
    img = img.convert('L')

    pixels = img.getdata()

    # replace each pixel with a character from array
    chars = ["B","S","#","&","@","$","%","*","!",":"," "]
    new_pixels = [chars[pixel//25] for pixel in pixels]
    new_pixels = ''.join(new_pixels)

    # split string of chars into multiple strings of length equal to new width and create a list
    new_pixels_count = len(new_pixels)
    ascii_image = [new_pixels[index:index + new_width] for index in range(0, new_pixels_count, new_width)]
    ascii_image = "\n".join(ascii_image)
    sys.stdout.write(ascii_image + "\n")

class HttpConfig:
    def __init__(self, img_base64_str):
        self.httpMethod = "POST"
        self.requstURL = "/v1/mnist_test"

        self.headerdata = {
            "Content-Type": "application/json"
        }

        self.test_data = {
            "image_base64": img_base64_str
        }

        self.body = json.dumps(self.test_data)

def DoMnistInfer(host, img_path, PrintRequest =  False):
    o = urlparse('//' + host)

    with open(img_path, 'rb') as fp:
        base64_data = base64.b64encode(fp.read())
        img_base64_str = str(base64_data, encoding='utf8')

    http_config = HttpConfig(img_base64_str)

    http_config.hostIP = o.hostname
    http_config.Port = o.port

    if PrintRequest:
        print("-- Request Body:")
        print(http_config.body)

    conn = http.client.HTTPConnection(host=http_config.hostIP, port=http_config.Port)
    try:
        conn.request(method=http_config.httpMethod, url=http_config.requstURL, body=http_config.body,
                headers=http_config.headerdata)
    except  Exception as e:
        print("Connect to " + host + " failed, reaseon: ")
        print(e)
        print("Please check the service status, or use '-host [ip:port]' to specify the address of the service.")
        return 1

    print("-- Response:")
    response = conn.getresponse().read().decode()
    print(response)
    return 0

def Extract(download):
    if os.path.exists(mnist_image_path):
        return
    
    if download:
        urllib.request.urlretrieve("http://download.modelbox-ai.com/test/mnist-image.tar.gz", "mnist-image.tar.gz")

    os.system("tar -C "+ str(source_dir) + " -xf mnist-image.tar.gz")

def main():
    descStr = "This program is used to test mnist inference."
    parser = argparse.ArgumentParser(description=descStr)
    parser.add_argument('-id', dest='TestNum', help="Test image id.", required=False)
    parser.add_argument('-download', dest='DownLoad', help="download full test image from modelbox.", required=False, default=False, action='store_true')
    parser.add_argument('-print-request', dest='PrintRequest', help="print request json body.", required=False, default=False, action='store_true')
    parser.add_argument('-host', dest='Host', default="127.0.0.1:8190", help="set host and port.", required=False)

    args = parser.parse_args()
    TestNum = 0

    Extract(args.DownLoad)

    if args.TestNum:
        TestNum = args.TestNum
    else:
        list = os.listdir(mnist_image_path)
        TestNum = random.randint(0, len(list) - 1)

    img_path = mnist_image_path + "/test_" + str(TestNum) + ".bmp"

    if not os.path.exists(img_path):
        print("image " + img_path + " not exists")
        return 1

    Host = args.Host
    PrintRequest = args.PrintRequest

    print("-- Image ID: " + str(TestNum) + ", Image: " )
    DisplayImage(img_path)
    print("-- Connect to " + Host)
    return DoMnistInfer(Host, img_path, PrintRequest)

if __name__ == "__main__":
    # execute only if run as a script
    main()
