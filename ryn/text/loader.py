# -*- coding: utf-8 -*-


import json
import sqlite3

from ryn.common import helper
from ryn.common import logging

from functools import partial
from contextlib import closing
from dataclasses import dataclass

from tqdm import tqdm as _tqdm

from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator


log = logging.get("text.loader")


tqdm = partial(_tqdm, ncols=80)


class Selector:
    def by_entity_id(self, entity_id: int):
        raise NotImplementedError()

    def by_entity(self, entity: str):
        raise NotImplementedError()


class Loader:
    def __enter__(self) -> Selector:
        raise NotImplementedError()

    def __exit__(self, *_):
        raise NotImplementedError()


# JSON --------------------


class JSONLoader(Loader):
    """

    Load a json


    The JSON file need to look like this:

    "<ENTITY>": {
        "description": "<DESCRIPTION"
    }

    And also it is required to provide an id mapping:
    { "<ENTITY>": <ID> }

    """

    @helper.notnone
    def __init__(self, fname: str = None, id2ent: Dict[int, str] = None):
        self.id2ent = id2ent

        path = helper.path(fname, exists=True, message="loading {path_abbrv}")
        with path.open(mode="r") as fd:
            raw = json.load(fd)

        self.ent2desc = {}
        for entity, data in raw.items():
            description = data["description"]
            if not description:
                self.ent2desc[entity] = None
            else:
                description = description.capitalize() + "."
                self.ent2desc[entity] = [description]

        assert all(ent in self.ent2desc for ent in id2ent.values())

    class JSONSelector(Selector):
        def __init__(
            self, ent2desc: Dict[int, List[str]], id2ent: Dict[int, str]
        ):
            self.ent2desc = ent2desc
            self.id2ent = id2ent

        def by_entity_id(self, entity_id: int):
            return self.ent2desc[self.id2ent[entity_id]]

        def by_entity(self, entity: str):
            return self.ent2desc[entity]

    def __enter__(self):
        return JSONLoader.JSONSelector(
            ent2desc=self.ent2desc, id2ent=self.id2ent
        )

    def __exit__(self, *_):
        pass


# SQLITE --------------------


@helper.notnone
def load_sqlite(
    *,
    database: str = None,
    batch_size: int = None,
) -> Generator[Tuple[str], None, None]:
    """

    Load text from a sqlite database (context_db)

    Schema must be like this (v4+):

    TABLE contexts:
        entity INT,
        entity_label TEXT,
        page_title TEXT,
        context TEXT,
        masked_context TEXT

    """
    query = "SELECT entity, context FROM contexts"

    with sqlite3.connect(database) as conn:
        with closing(conn.cursor()) as c:
            c.execute(query)

            res = [None]
            while len(res):
                res = c.fetchmany(batch_size)
                yield res


class SQLite(Loader):
    """

    Load text from a sqlite database

    Schema must be like this (v4+):

    TABLE contexts:
        entity INT,
        entity_label TEXT,
        page_title TEXT,
        context TEXT,
        masked_context TEXT

    "entity_label" is the mention

    """

    DB_NAME = "contexts"

    # ---

    COL_ID: int = "id"
    COL_ENTITY: int = "entity"
    COL_LABEL: str = "entity_label"
    COL_MENTION: str = "mention"
    COL_CONTEXT: str = "context"
    COL_CONTEXT_MASKED: str = "masked_context"

    # ---

    @dataclass
    class SQLSelector(Selector):

        conn: sqlite3.Connection
        cursor: sqlite3.Cursor

        def by_entity_id(self, entity_id: int):
            query = (
                "SELECT "
                f"{SQLite.COL_ENTITY}, "
                f"{SQLite.COL_MENTION}, "
                f"{SQLite.COL_CONTEXT}, "
                f"{SQLite.COL_CONTEXT_MASKED} "
                f"FROM {SQLite.DB_NAME} "
                f"WHERE {SQLite.COL_ENTITY}=?"
            )

            params = (entity_id,)

            self.cursor.exectute(query, params)
            return self.cursor.fetchall()

        def by_entity(self, entity: str):
            query = (
                "SELECT "
                f"{SQLite.COL_ENTITY}, "
                f"{SQLite.COL_MENTION}, "
                f"{SQLite.COL_CONTEXT}, "
                f"{SQLite.COL_CONTEXT_MASKED} "
                f"FROM {SQLite.DB_NAME} "
                f"WHERE {SQLite.COL_ENTITY}=?"
            )

            params = (entity,)

            self.cursor.execute(query, params)
            return self.cursor.fetchall()

    # ---

    def __init__(self, *, database: str = None, to_memory: bool = False):
        log.info(f"connecting to database {database}")

        if to_memory:
            log.info("copying database to memory")
            self._conn = sqlite3.connect(":memory:")
            self._cursor = self._conn.cursor()

            log.info(f"opening {database}")
            with sqlite3.connect(database) as con:
                for sql in con.iterdump():
                    self._cursor.execute(sql)

        else:
            log.info("accessing database from disk")
            self._conn = sqlite3.connect(database)
            self._cursor = self._conn.cursor()

    def __enter__(self):
        return SQLite.SQLSelector(conn=self._conn, cursor=self._cursor)

    def __exit__(self, *_):
        self._conn.close()
