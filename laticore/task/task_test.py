import unittest
import json

from laticore.task.task import Task, obj

class TestTask(unittest.TestCase):
    def setUp(self):
        self.test_dict = {
            "a":"b",
            "c": {
                "d": "e",
                "f" : ["g","a","b"],
            },
        }
        self.test_json = json.dumps(self.test_dict)

    def test_task_init(self):
        task = Task(self.test_json)
        self.assertTrue(isinstance(task.as_json, str))
        self.assertTrue(isinstance(task.as_dict, dict))
        self.assertEqual(self.test_dict["a"], task.as_dict["a"])
        self.assertEqual(self.test_json, task.as_json)
        self.assertEqual(self.test_dict["a"], task.a)
        self.assertEqual(self.test_dict["c"]["d"], task.c.d)
