# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import bokeh.plotting as plt
import sys


def show(file):
    plt.output_file('lifetime.html')
    fig1 = plt.figure(width=1600, height=800)
    with open(file, 'r') as f:
        f.readline() # skip first line
        while True:
            items = f.readline().split(' ')
            if len(items) != 5:
                break
            left = float(items[3])
            right = float(items[4])
            bottom = float(items[1])
            top = float(items[2])

            fig1.quad(top=top, bottom=bottom, left=left, right=right, line_color="black", line_width=2)
    plt.show(fig1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show liftime plot.')
    parser.add_argument('file', type=str, help='lifetime file')

    args = parser.parse_args()
    show(args.file)
