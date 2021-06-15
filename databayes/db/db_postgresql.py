import psycopg2
import pydantic
import typing
import sys
import logging

from .db_base import DBBase, DBConfigBase

import pkg_resources

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: F401


# TODO !!!!
class DBPostgreSQL(DBBase):

    def connect(self, logging=logging, **params):
        """ Connect to the PostgreSQL database server """

        try:
            # connect to the PostgreSQL server
            logging.info(f'Connecting to the PostgreSQL database {self.name}')
            self.db = psycopg2.connect(**self.config.dict())
        except (Exception, psycopg2.DatabaseError) as error:
            logging.error(error)
            sys.exit(1)
        logging.info("Connection successful")
        # self.db = pymongo.MongoClient(host=self.config["host"],
        #                               port=self.config["port"],
        #                               serverSelectionTimeoutMS=serverSelectionTimeoutMS,
        #                               **params)

        # self.db.server_info()

    def prepare_and_get_coll(self, endpoint, index=[]):
        db_name, coll_name = endpoint.split("/")

        db_coll = self.db[db_name][coll_name]
        if not(index is None):
            if not(isinstance(index, list)):
                index = [index]

            if len(index) > 0:
                db_coll.create_index([(idx, 1) for idx in index],
                                     unique=True)
        else:
            index = []

        return db_coll, index

    def size(self, endpoint,
             filter={},
             **params):

        db_coll, index = self.prepare_and_get_coll(endpoint)
        return db_coll.count_documents(filter=filter)

    def get(self, endpoint,
            filter={},
            projection=[],
            limit=0,
            keep_mongo_id=False,
            **params):

        db_coll, index = self.prepare_and_get_coll(endpoint)

        mongo_projection = {var: 1 for var in projection}
        mongo_projection.update({"_id": int(keep_mongo_id)})

        data_cs = db_coll.find(filter=filter,
                               projection=mongo_projection).limit(limit)

        return list(data_cs)

    def replace(self, endpoint,
                data=[],
                index=[],
                set_on_insert_data={},
                logger=None,
                **params):

        db_coll, index = self.prepare_and_get_coll(endpoint, index)

        filter = {idx: data[idx] for idx in index}

        if isinstance(data, list):
            query_res_info = \
                [db_coll.replace_one(
                    filter,
                    dict(d, **set_on_insert_data),
                    upsert=True) for d in data]
        else:
            query_res_info = \
                [db_coll.replace_one(filter,
                                     dict(data, **set_on_insert_data),
                                     upsert=True)]

        if not(logger is None):
            # TODO: STOP USING THIS USE LOG_DB_OPS INSTEAD
            logger.debug(mongoDataBackend.format_update_res(db_coll,
                                                            query_res_info))

    def update(self, endpoint,
               data=[],
               index=[],
               set_on_insert_data={},
               logger=None,
               **params):

        db_coll, index = self.prepare_and_get_coll(endpoint, index)

        set_query = {}
        if len(set_on_insert_data) > 0:
            set_query.update({"$setOnInsert": set_on_insert_data})

        if isinstance(data, list):
            update_res = \
                [db_coll.update_one(
                    {idx: d[idx] for idx in index},
                    dict({"$set": d}, **set_query),
                    upsert=True) for d in data]
        else:
            update_res = \
                [db_coll.update_one({idx: data[idx] for idx in index},
                                    dict({"$set": data}, **set_query),
                                    upsert=True)]

        if not(logger is None):
            # TODO: STOP USING THIS USE LOG_DB_OPS INSTEAD
            logger.debug(mongoDataBackend.format_update_res(db_coll,
                                                            update_res))

    def put(self, endpoint,
            data=[],
            index=[],
            logger=None,
            **params):

        res_dict = {"data_name": endpoint,
                    "ops_type": "insert-update",
                    "ops": dict([
                        ("nb_data_processed", len(data)),
                        ("nb_inserts", 0),
                        ("nb_updates", 0)])}

        db_coll, index = self.prepare_and_get_coll(endpoint, index)

        # if has_index:
        #     # Drop duplicates based on index before inserting
        #     df.drop_duplicates(subset=data_index, inplace=True)
        try:
            if isinstance(data, list):
                if len(data) == 0:
                    return res_dict

                ops_res_cur = \
                    db_coll.insert_many(data, ordered=False)

                res_dict["ops"]["nb_inserts"] += len(ops_res_cur.inserted_ids)

            else:
                ops_res_cur = \
                    db_coll.insert_one(data)

                res_dict["ops"]["nb_inserts"] += ops_res_cur.acknowledged

        except Exception as err:
            if not(logger is None):
                logger.error("Problem occurred inserting data in endpoint {} : {}"
                             .format(endpoint, err))

        mongoDataBackend.log_db_ops(res_dict,
                                    logger=logger)

        return res_dict

    def delete(self, endpoint,
               filter={},
               logger=None,
               **params):

        res_dict = {"data_name": endpoint,
                    "ops_type": "delete",
                    "ops": dict([
                        ("nb_deletions", 0)])}

        db_coll, index = self.prepare_and_get_coll(endpoint)

        ops_res_cur = db_coll.delete_many(filter=filter)

        res_dict["ops"]["nb_deletions"] = ops_res_cur.deleted_count

        mongoDataBackend.log_db_ops(res_dict,
                                    logger=logger)

        return res_dict

    @staticmethod
    def log_db_ops(res_dict, logger=None):
        log_msg_list = []

        log_msg_list.append("Data source '{}' - '{}' operation".format(res_dict["data_name"],
                                                                       res_dict["ops_type"]))
        log_msg_list.append("in {:.3} seconds".format(res_dict["etime"])
                            if "etime" in res_dict.keys() else "")

        log_msg = " ".join(log_msg_list)
        if not(logger is None):
            logger.info(log_msg)
            [logger.info("> {}: {}".format(ops, n))
             for ops, n in res_dict["ops"].items()]

    # TODO: STOP USING THIS USE LOG_DB_OPS INSTEAD
    @staticmethod
    def format_update_res(coll, res):
        if isinstance(res, list):
            update_keys = ["n", "nModified", "ok"]
            res_bis = {k: sum([r.raw_result[k] for r in res])
                       for k in update_keys}
            res_bis["upserted"] = [r.raw_result["upserted"]
                                   for r in res if "upserted" in r.raw_result.keys()]
            res_bis["nUpserted"] = len(res_bis["upserted"])
        else:
            res_bis = res.raw_result.copy()
            res_bis["nUpserted"] = 1 if "upserted" in res_bis.keys() else 0

        return "> collection {}: {:.0f} update request(s) correctly processed - items inserted: {:.0f} - items updated: {:.0f}" \
            .format(coll.name,
                    res_bis["ok"],
                    res_bis["nUpserted"],
                    res_bis["nModified"])
