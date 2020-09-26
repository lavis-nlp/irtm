# -*- coding: utf-8 -*-


import sqlite3

from ryn.common import helper
from ryn.common import logging

from functools import partial
from contextlib import closing
from dataclasses import dataclass

from tqdm import tqdm as _tqdm

from typing import Tuple
from typing import Generator


log = logging.get('text.loader')


tqdm = partial(_tqdm, ncols=80)


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
      entity INTEGER
      entity_label TEXT
      context TEXT

    "entity_label" is the mention

    """

    DB_NAME = 'contexts'

    # ---

    COL_ID = 'id'
    COL_ENTITY = 'entity'
    COL_MENTION = 'entity_label'
    COL_CONTEXT = 'context'

    # ---

    @dataclass
    class Selector:

        conn: sqlite3.Connection
        cursor: sqlite3.Cursor

        def by_entity_id(self, entity_id: int, count: bool = False):
            query = (
                'SELECT '
                f'{SQLite.COL_ENTITY}, '
                f'{SQLite.COL_MENTION}, '
                f'{SQLite.COL_CONTEXT} '

                f'FROM {SQLite.DB_NAME} '
                f'WHERE {SQLite.COL_ENTITY}=?')

            params = (entity_id, )

            self.cursor.exectute(query, params)
            return self.cursor.fetchall()

        def by_entity(self, entity: str, count: bool = False):
            query = (
                'SELECT '
                f'{SQLite.COL_ENTITY}, '
                f'{SQLite.COL_MENTION}, '
                f'{SQLite.COL_CONTEXT} '

                f'FROM {SQLite.DB_NAME} '
                f'WHERE {SQLite.COL_ENTITY}=?')

            params = (entity, )

            self.cursor.execute(query, params)
            return self.cursor.fetchall()

    # ---

    def __init__(self, *, database: str = None, to_memory: bool = False):
        log.info(f'connecting to database {database}')

        if to_memory:
            log.info('copying database to memory')
            self._conn = sqlite3.connect(':memory:')
            self._cursor = self._conn.cursor()

            log.info(f'opening {database}')
            with sqlite3.connect(database) as con:
                # bar = partial(tqdm,  desc='copy to memory', unit=' statements')  # noqa
                for sql in con.iterdump():
                    self._cursor.execute(sql)

        else:
            log.info('accessing database from disk')
            self._conn = sqlite3.connect(database)
            self._cursor = self._conn.cursor()

    def __enter__(self):
        return SQLite.Selector(conn=self._conn, cursor=self._cursor)

    def __exit__(self, *_):
        self._conn.close()
