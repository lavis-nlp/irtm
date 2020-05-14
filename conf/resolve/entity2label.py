#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse


def main(args):

    m2i = {}  # mid -> idx

    with open(args.e2i, mode='r') as fd:
        fd.readline()

        for line in fd:
            mid, idx = line.split()
            m2i[mid] = int(idx)

    m2l = {}  # mid -> label

    with open(args.e2w, mode='r') as fd:
        data = json.load(fd)
        for mid in data:
            m2l[mid] = data[mid]['label']

    with open(args.out, mode='w') as fd:
        fd.write(f'{len(m2i)}\n')

        for mid, idx in m2i.items():
            fd.write(f'{m2l.get(mid, mid)} {idx}\n')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'e2i', type=str,
        help='the entity2id file')

    parser.add_argument(
        'e2w', type=str,
        help='the entity2wikidata file')

    parser.add_argument(
        'out', type=str,
        help='target text file')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
