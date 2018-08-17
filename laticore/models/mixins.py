import os
import bson
import h5py
import redis
import pymongo
import tensorflow as tf
from datetime import datetime, timedelta
from keras.models import load_model

from laticore.models.exceptions import NullModelException, ModelNotFoundException

class ModelLockMixin(object):
    """
    Helper class that handles locking/unlocking of models for safe writes
    """
    def __init__(self):
        """
        Args:
            redis_session (redis.Redis, required): redis session
            task_id (str, required): task id string
            tenant (str, required): name of tenant
            lock_expiration_minutes (int, required): how many minutes until the
                lock automatically expires
            lock_wait_timeout_seconds (int, required): how many seconds to wait to
                acquire the lock if it's owned by somebody else
        """
        # instance of redis.Lock
        self._redis_lock = None

        # whether this instance actually owns the lock for the model
        self.locked = False

    @property
    def locked(self) -> bool:
        return self._locked

    @locked.setter
    def locked(self, val):
        if not isinstance(val, bool):
            raise TypeError("locked must be of type bool")
        self._locked = val

    @property
    def lock_key(self) -> str:
        return self._lock_key

    @lock_key.setter
    def lock_key(self, val):
        if not isinstance(val, str):
            raise TypeError("lock_key must be of type str")
        self._lock_key = val

    def fmt_lock_key(self, tenant:str, task_id:str) -> str:
        """
        Combines tenant and task_id into a single string to be used as a lock key

        Args:
            tenant (str, required)
            task_id (str, required)

        Returns:
            lock_key (str)
        """
        return "%s-%s" % (tenant, task_id)

    def lock(self, redis_session: redis.Redis, task_id:str, tenant:str,
        lock_expiration_seconds:int, lock_wait_timeout_seconds:int, sleep_seconds:float = .1) -> bool:
        """
        Checks for the existence of a lock and creates one if possible,
        thereby setting self._locked to True
        """
        self.lock_key = self.fmt_lock_key(tenant, task_id)

        self._redis_lock = redis_session.lock(
            self.lock_key,
            timeout             = lock_expiration_seconds,
            sleep               = sleep_seconds,
            blocking_timeout    = lock_wait_timeout_seconds,
        )
        self.locked = self._redis_lock.acquire()

        return self.locked

    def unlock(self):
        """
        unlock verifies the lock is owned by the current instance and deletes
        it. Sets lock_acquired to False
        """
        try:
            self._redis_lock.release()
        except redis.exceptions.LockError as e:
            self.locked = False
            raise e
        else:
            self.locked = False

class ModelStorageMixin(object):
    """
    Helper class that handles fetching and storage of models from mongodb
    """

    def __init__(self, mongo_session:pymongo.MongoClient, collection:str,
        local_model_storage:str, task_id:str, tenant:str):
        """
        Args:
            mongo_session (pymongo.MongoClient, required): mongo session
            collection (str, required): collection where model is stored
            local_model_storage (str, required): directory where models will be temporarily written
            task_id (str, required): id of the task
            tenant (str, required): tenant name
        """

        self.mongo = mongo_session
        self.collection = collection
        self.local_model_storage = local_model_storage
        self.task_id = task_id
        self.tenant = tenant

        # raw database response for retrieving model from storage
        self.model_doc = None

        # whether the model has been loaded from storage
        self.model_loaded = False

        # need a random object id for tensorflow name scope
        self.tf_name_scope = str(bson.ObjectId())

    @property
    def model_path(self):
        return os.path.join(self.local_model_storage, self.task_id + ".h5")

    def exists_in_db(self) -> bool:
        """
        Checks whether model exists in storage

        Returns:
            bool
        """
        return self.mongo[self.tenant][self.collection].count({"task_id": bson.ObjectId(self.task_id)}) > 0

    def fetch_and_set_from_db(self):
        """
        Fetches model from storage and sets as instance attribute self._model
        """
        self.model_doc = self.mongo[self.tenant][self.collection].find_one({"task_id": bson.ObjectId(self.task_id)})

        if self.model_doc is None:
            raise ModelNotFoundException("Model not found in storage.")

        # write to file
        with open(self.model_path, 'wb') as f:
            f.write(self.model_doc["model"])

        # load model from file
        with tf.name_scope(self.tf_name_scope):
            self._model = load_model(self.model_path)
            self._model._make_predict_function()

    def save_new(self, earliest_metric_time:datetime):
        """
        Saves new model to the mongo. Includes earliest metric time in the record

        Performs an upsert operation because it's possible that the model record
        exists in the database and we're overwriting it. This can happen for
        a number of reasons. One common case is re-training. We re-train the model
        and then overwrite the existing record.

        Args:
            earliest_metric_time (datetime.datetime, required): earliest metric time the model was trained on

        Returns:
            result (pymongo.results.UpdateResult)
                attributes:
                    acknowledged
                    matched_count
                    modified_count
                    raw_result
                    upserted_id
        """
        if not self._model:
            raise NullModelException("self._model must be set in order to save it.")
        # save model to .h5 file locally
        self._model.save(self.model_path)

        with open(self.model_path, 'rb') as f:
            data = f.read()

        # since model is being newly saved we need to include the earliest_metric_time
        # field in the document so we know the timestamp of the verfy first metric
        # included in the training. This is so we can determine during future scaling
        # events if the model has seen sufficient training data (i.e. minTrainSize) to
        # take action
        update_doc = {
            "$set": {
                "last_updated": datetime.utcnow(),
                "model": data,
                "earliest_metric_time": earliest_metric_time,
            }
        }

        return self.mongo[self.tenant][self.collection].update_one(
            {"task_id": bson.ObjectId(self.task_id)},
            update_doc,
            upsert=True
        )

    def save_update(self) -> pymongo.results.UpdateResult:
        """
        Updates model in mongodb storage

        Returns:
            result (pymongo.results.UpdateResult)
                attributes:
                    acknowledged
                    matched_count
                    modified_count
                    raw_result
                    upserted_id
        """
        # save model to .h5 file locally
        self._model.save(self.model_path)

        with open(self.model_path, 'rb') as f:
            data = f.read()

        update_doc = {
            "$set": {
                "last_updated": datetime.utcnow(),
                "model": data,
            }
        }

        return self.mongo[self.tenant][self.collection].update_one(
            {"task_id": bson.ObjectId(self.task_id)},
            update_doc,
            upsert=True
        )
