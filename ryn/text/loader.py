# -*- coding: utf-8 -*-


import sqlite3

from contextlib import closing
from dataclasses import dataclass

from ryn.common import helper

from typing import Tuple
from typing import Generator


@helper.notnone
def load_sqlite(
        *,
        database: str = None,
        batch_size: int = None, ) -> Generator[Tuple[str], None, None]:
    """

    Load text from a sqlite database

    Schema must be like this:

    TABLE contexts:
      id INTEGER PRIMARY KEY AUTOINCREMENT
      entity TEXT
      context TEXT

    """
    query = 'SELECT entity, context FROM contexts'

    with sqlite3.connect(database) as conn:
        with closing(conn.cursor()) as c:
            c.execute(query)

            res = [None]
            while len(res):
                res = c.fetchmany(batch_size)
                yield res


class SQLite:
    """

    Load text from a sqlite database

    Schema must be like this:

    TABLE contexts:
      id INTEGER PRIMARY KEY AUTOINCREMENT
      entity TEXT
      context TEXT

    """

    DB_NAME = 'contexts'

    # ---

    COL_ID = 'id'
    COL_RYN_ID = 'ryn_id'
    COL_ENTITY = 'entity'
    COL_MENTION = 'mention'
    COL_CONTEXT = 'context'

    # ---

    @dataclass
    class Selector:

        conn: sqlite3.Connection
        cursor: sqlite3.Cursor

        def by_ryn_id(self, ryn_id: int, count: bool = False):
            raise NotImplementedError

        def by_entity(self, entity: str, count: bool = False):
            query = (
                f'SELECT {SQLite.COL_ENTITY}, {SQLite.COL_CONTEXT} '
                f'FROM {SQLite.DB_NAME} '
                f'WHERE {SQLite.COL_ENTITY}=?')

            params = (entity, )

            self.cursor.execute(query, params)
            return self.cursor.fetchall()

    # ---

    def __init__(self, *, database: str = None):
        self._conn = sqlite3.connect(database)
        self._cursor = self._conn.cursor()

    def __enter__(self):
        return SQLite.Selector(conn=self._conn, cursor=self._cursor)

    def __exit__(self, *_):
        self._conn.close()
