import os
import bson
import redis
import shutil
import pymongo
import unittest
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Dense
from keras import layers

import laticore.models.test_config as config
from laticore.models.mixins import *

class TestModelLockMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.redis_session = redis.StrictRedis(**config.REDIS)

    def setUp(self):
        self.mixin = ModelLockMixin()

    def tearDown(self):
        if self.mixin.locked:
            try:
                self.mixin.unlock()
            except Exception as e:
                pass

    def test_fmt_lock_key(self):
        self.assertEqual(self.mixin.fmt_lock_key("foo","bar"), "foo-bar")

    def test_lock(self):
        result = self.mixin.lock(
            redis_session = self.redis_session,
            task_id = "foo",
            tenant = "bar",
            lock_expiration_seconds = 5,
            lock_wait_timeout_seconds = 1,
        )

        self.assertTrue(result)

    def test_unlock(self):
        locked = self.mixin.lock(
            redis_session = self.redis_session,
            task_id = "foo",
            tenant = "bar",
            lock_expiration_seconds = 5,
            lock_wait_timeout_seconds = 1,
        )
        self.assertTrue(locked)
        self.mixin.unlock()
        self.assertFalse(self.mixin.locked)


class TestModelStorageMixin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mongo = pymongo.MongoClient(**config.MONGO)

        if not os.path.exists("./model_storage"):
            os.mkdir("./model_storage")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./model_storage"):
            shutil.rmtree("./model_storage")

    def model_factory(self):
        model = Sequential()
        model.add(layers.LSTM(
            50,
            input_shape = (20,4),
            activation = "linear",
        ))
        model.add(Dense(units = 20,))
        model.compile(
            loss = "binary_crossentropy",
            optimizer = "adam",
        )
        return model

    def setUp(self):
        self.mixin = ModelStorageMixin(
            mongo_session = self.mongo,
            collection = "test_collection",
            local_model_storage = "./model_storage",
            task_id = str(bson.ObjectId()),
            tenant = "test_tenant_" + str(bson.ObjectId()),
        )

    def tearDown(self):
        self.mongo.drop_database(self.mixin.tenant)

    def test_save_new(self):
        self.mixin._model = self.model_factory()
        result = self.mixin.save_new(earliest_metric_time=datetime.now())
        self.assertTrue(result.acknowledged)
        self.assertTrue(result.upserted_id != None)

    def test_save_update(self):
        self.mixin._model = self.model_factory()
        result = self.mixin.save_new(earliest_metric_time=datetime.now())
        self.assertTrue(result.upserted_id != None)

        result = self.mixin.save_update()
        self.assertTrue(result.acknowledged)
        self.assertEqual(1, result.matched_count)
        self.assertEqual(1, result.modified_count)

    def test_fetch_and_set_from_db(self):
        self.mixin._model = self.model_factory()
        result = self.mixin.save_new(earliest_metric_time=datetime.now())
        self.assertTrue(result.upserted_id != None)

        mixin2 = ModelStorageMixin(
            mongo_session = self.mongo,
            collection = "test_collection",
            local_model_storage = "./model_storage",
            task_id = self.mixin.task_id,
            tenant = self.mixin.tenant,
        )
        mixin2.fetch_and_set_from_db()
        self.assertTrue(mixin2.model_doc != None)
        self.assertTrue(mixin2._model != None)

    def test_exists_in_db(self):
        self.mixin._model = self.model_factory()
        result = self.mixin.save_new(earliest_metric_time=datetime.now())
        self.assertTrue(result.upserted_id != None)

        self.assertTrue(self.mixin.exists_in_db())
