import bson
import json
import boto3

class obj(object):
    """
    Special class, that when instantited, recursively turns the provided dictionary
    into instance attributes.

    Thus, a dictionary schema like this

    d = {
        "a" : "b",
        "c" : "d",
        "e" : {
            "f" : "g",
            "h" : "i",
            "j" : ["k", "l", "m"],
        },
    }

    becomes an object where:

    d.a -> "b"
    d.e.j -> ["k", "l", "m"]
    """
    def __init__(self, d: dict):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

class Task(obj):
    def __init__(self, task_json: str):
        self.as_json = task_json
        self.as_dict = json.loads(task_json)
        super().__init__(self.as_dict)

    @property
    def as_json(self):
        return self._as_json

    @as_json.setter
    def as_json(self, val):
        if not isinstance(val, str):
            raise TypeError("task.json must be instance of str")
        self._as_json = val

    @property
    def as_dict(self):
        return self._as_dict

    @as_dict.setter
    def as_dict(self, val):
        if not isinstance(val, dict):
            raise TypeError("task.dictionary must be instance of dict")
        self._as_dict = val
