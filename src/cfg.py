# src/cfg.py

import json

class DictToObj:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictToObj(v)
            setattr(self, k, v)


class Config:
    def __init__(self, path):
        self.path = path

    def load(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if isinstance(value, dict):
                value = DictToObj(value)
            setattr(self, key, value)

        return self