import re
from enum import IntFlag, auto
import os
from typing import Tuple, List
import io

ITEM_PATTERN = re.compile(
    r"^DataItem\(\d+, \"(\w+)\", (True|False), (0\.0), (0\.0), (0\.0)\),", re.RegexFlag.MULTILINE)


class Status(IntFlag):
    find_titile = auto()
    find_time = auto()


def find_titile(line: str) -> str:
    title_pattern = re.compile(r"^\d+:([a-zA-Z0-9_.-]+)\s:\s")
    match = title_pattern.match(line)
    if match is None:
        return None
    return match.group(1)


def find_time(line: str) -> Tuple[str, str]:
    time_pattern = re.compile(r"^\|(\w+)\s+\|(\d+|\d+.\d+)\s+\|(\d+|\d+.\d+)\s+\|")
    match = time_pattern.match(line)
    if match is None:
        return None
    return match.group(2), match.group(3)


def find_items(info_path: str) -> int:
    if not os.path.exists(info_path):
        return -1
    context = None
    with open(info_path, 'r') as f:
        context = f.read()
    return len(ITEM_PATTERN.findall(context))


def update_items(info_path: str, times: List[Tuple[str, str]]):
    if not os.path.exists(info_path):
        return -1
    context = None
    with open(info_path, 'r') as f:
        context = f.read()

    cnt = {'i': 0}

    def update(match: re.Match):
        i = cnt['i']
        time = times[i]
        new = f'DataItem({i}, \"{match.group(1)}\", {match.group(2)}, {time[0]}, {time[1]}, {float(time[1])-float(time[0]):.6f}),'
        cnt['i'] += 1
        return new

    new_context = ITEM_PATTERN.sub(update, context)
    with open(info_path, 'w') as f:
        f.write(new_context)


def update_trace_info(infer_result: str, info_file: str):
    status = Status.find_titile
    title = None
    item_num = -1
    times = []

    buf = io.StringIO(infer_result)
    while True:
        line = buf.readline()
        if not line:
            break

        if status == Status.find_titile:
            title = find_titile(line)
            if title:
                status = Status.find_time
                item_num = find_items(info_file)
                if item_num == -1 or item_num == 0:
                    item_num = -1
                    status = Status.find_titile
                continue

        if status is Status.find_time:
            time = find_time(line)
            if time:
                times.append(time)
            if (len(times) == item_num):
                update_items(info_file, times)
                times.clear()
                status = Status.find_titile
            continue
