import argparse
import json
import pandas as pd


def json2md(json_file):
    json_list = []
    with open(json_file, 'r') as f:
        json_list = json.load(f)

    json_list = sorted(json_list, key=lambda d: d['case'])
    df = pd.DataFrame.from_records(json_list)
    md = df.to_markdown()
    md_file = json_file.split('/')[-1].split('.')[0] + '.md'

    with open(md_file, 'w') as f:
        f.write(md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="json2md")
    parser.add_argument("--json", help='json file', type=str)
    args = parser.parse_args()
    json2md(args.json)
