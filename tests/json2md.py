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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import argparse
import json
import os
from pathlib import Path


def json2md(json_file):
    file_list = []
    for path in Path(os.path.dirname(json_file)).glob(f'{json_file}*.json'):
        file_list.append(path)

    json_list = []
    for f in file_list:
        with open(f, 'r') as f:
            json_list.extend(json.load(f))
    assert(len(json_list) > 0)

    # generate dict after sorting
    json_list = sorted(json_list, key=lambda d: (d['priority'], d['kind'], d['model']))
    dict = {}
    for e in json_list:
        kind = e['kind']
        if kind not in dict:
            dict[kind] = []
        dict[kind].append(e)

    # generate html table
    md = '<table>\n'

    # table head
    md += '\t<tr>\n'
    for key in json_list[0]:
        if key != 'priority':
            md += f'\t\t<th>{key}</th>\n'
    md += '\t</tr>\n'

    # table row
    for value in dict.values():
        length = len(value)
        for i in range(length):
            md += '\t<tr>\n'
            if i == 0:
                for k, v in value[i].items():
                    if k == 'kind':
                        md += f'\t\t<td rowspan=\'{length}\'>{v}</td>\n'
                    elif k != 'priority':
                        md += f'\t\t<td>{v}</td>\n'
            else:
                for k, v in value[i].items():
                    if k != 'kind' and k != 'priority':
                        md += f'\t\t<td>{v}</td>\n'
            md += '\t</tr>\n'

    md += '</table>\n'

    md_file = os.path.splitext(json_file)[0] + '.md'
    with open(md_file, 'w') as f:
        f.write(md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="json2md")
    parser.add_argument("--json", help='json file or json file prefix', type=str)
    args = parser.parse_args()
    json2md(args.json)
