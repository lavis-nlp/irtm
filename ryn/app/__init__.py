# -*- coding: utf-8 -*-


from dataclasses import dataclass


@dataclass
class Context:

    def run(self):
        raise NotImplementedError()
