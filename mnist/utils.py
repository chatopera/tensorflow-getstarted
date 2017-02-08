# encoding: UTF-8
# Copyright 2017 Hai Liang Wang
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

import os
import socket
from time import localtime, strftime
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')


def gen_model_tagname(prefix='tensorflow'):
    return "%s.mnist.%s.%s" % (prefix,
                               socket.gethostname(),
                               strftime("%Y%m%d.%H%M%S", localtime()))


def gen_model_save_dir(prefix='tensorflow'):
    return os.path.join(save_dir, gen_model_tagname(prefix))


def main():
    print(save_dir)
    print(gen_model_tagname())
    print(gen_model_save_dir())

if __name__ == '__main__':
    main()
