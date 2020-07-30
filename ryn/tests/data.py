from ryn.graphs.graph import frozendict
from ryn.graphs.graph import GraphImport

import pickle
import pathlib
import unittest
from dataclasses import FrozenInstanceError


class GraphImportTests(unittest.TestCase):

    def test_initialization(self):
        s = GraphImport(
            triples=[(0, 1, 2), (3, 4, 5), (1, 3, 2)],
            ents={0: 'zero', 3: 'three'},
            rels={5: 'five'}
        )

        self.assertIn((0, 1, 2), s.triples)
        self.assertIn((3, 4, 5), s.triples)
        self.assertIn((1, 3, 2), s.triples)

        self.assertEqual(s.ents[0], 'zero')
        self.assertEqual(s.ents[1], '1')
        self.assertEqual(s.ents[3], 'three')
        self.assertEqual(s.ents[4], '4')

        self.assertEqual(s.rels[2], '2')
        self.assertEqual(s.rels[5], 'five')

    def test_join(self):
        s1 = GraphImport(triples=[(0, 1, 2), (3, 4, 5)])

        s2 = GraphImport(
            triples=[(0, 1, 2), (1, 3, 2)],
            rels={2: 'two'}, ents={5: 'five'})

        s1.join(s2)

        self.assertEqual(len(s1.triples), 3)
        self.assertIn((0, 1, 2), s1.triples)
        self.assertIn((3, 4, 5), s1.triples)
        self.assertIn((1, 3, 2), s1.triples)

        self.assertEqual(s1.rels[2], 'two')
        self.assertEqual(s2.ents[5], 'five')

    def test_immutability(self):
        s = GraphImport(triples=[(0, 0, 0)])

        with self.assertRaises(FrozenInstanceError):
            s.triples = [(1, 1, 1)]

        with self.assertRaises(FrozenInstanceError):
            s.rels = {0: 'zero'}

        with self.assertRaises(FrozenInstanceError):
            s.ents = {0: 'zero'}

        with self.assertRaises(FrozenInstanceError):
            s.rels[0] = 'zero'

        with self.assertRaises(FrozenInstanceError):
            s.ents[0] = 'zero'

        with self.assertRaises(AttributeError):
            s.triples.add(3)

    def test_pickling(self):
        d1 = frozendict([('1', '2'), ('3', '4')])

        p_pkl = pathlib.Path('.tmp.pkl')
        if p_pkl.exists():
            self.fail(f'There is a {p_pkl}')

        exc = None

        try:
            pass
            with p_pkl.open(mode='wb') as fd:
                pickle.dump(d1, fd)

            with p_pkl.open(mode='rb') as fd:
                d2 = pickle.load(fd)

            self.assertEqual(d1, d2)

            with self.assertRaises(FrozenInstanceError):
                d2['4'] = '4'

            with self.assertRaises(FrozenInstanceError):
                d2['3'] = '3'

        except Exception as e:
            exc = e

        finally:
            if p_pkl.exists():
                p_pkl.unlink()

            if exc is not None:
                raise exc
