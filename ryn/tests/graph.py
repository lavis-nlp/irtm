# -*- coding: utf-8 -*-


from ryn.graphs.graph import Graph
from ryn.graphs.graph import GraphImport
from ryn.graphs.graph import PartitionConstraintError

import unittest


"""
The following more complex graphs are used for most of the tests.
They contains the most important relation types and suitable depth
for more complex subsampling and path-finding algorithms.

Take a sheet of paper and draw them :)

"""
test_graph_rels = GraphImport(
    triples=(
        (0, 1, 0),
        (0, 2, 0),
        (0, 2, 3),
        (0, 3, 2),
        (3, 0, 2),
        (2, 2, 1),
        (0, 4, 3),
        (4, 2, 3),
        (4, 5, 2),
        (5, 4, 2),
        (5, 6, 0),
        (6, 7, 0),
        (6, 3, 0),
    ),
    rels={
        0: 'multi',
        1: 'reflexive',
        2: 'symmetric',
        3: 'transitive',
    },
    ents={
        0: 'center',
    }
)


test_graph_paths = GraphImport(
    triples=(
        (0, 1, 0),
        (1, 0, 0),
        (2, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
        (2, 4, 0),
        (2, 4, 1),
        (2, 5, 0),
        (4, 4, 0),
        (4, 5, 0),
        (5, 4, 0),
        (5, 6, 0),
    )
)


# ---


class BasicGraphTests(unittest.TestCase):

    def test_empty(self):
        g = Graph(name='test_empty')

        self.assertEqual(g.name, 'test_empty')
        self.assertEqual(len(g.source.rels), 0)

        self.assertEqual(g.source.triples, set())
        self.assertEqual(g.source.rels, {})
        self.assertEqual(g.source.ents, {})

        g.str_stats

    def test_one(self):
        """

        Tests on a simple graph with one reflexive node.

        """

        triples = [(0, 0, 0)]
        s = GraphImport(triples, )
        one = Graph(name='one_reflexive', source=s)

        self.assertEqual(one.source.triples, set(triples))

        self.assertEqual(len(one.source.rels), 1)
        self.assertEqual(one.nx.number_of_edges(), 1)
        self.assertEqual(one.nx.number_of_nodes(), 1)
        self.assertEqual(one.nx.nodes[0], {'label': '0'})
        self.assertEqual(one.nx.edges[0, 0, 0], {'rid': 0, 'label': '0'})

    def test_two(self):
        """

        Tests on a simple graph with two fully connected nodes.
        Also provide custom labels for some.

        """
        triples = [(0, 1, 0), (1, 0, 0)]
        s = GraphImport(triples=triples,
                        rels={0: 'custom_rel'},
                        ents={0: 'custom_ent'})

        two = Graph(name='two', source=s)

        self.assertEqual(two.source.triples, set(triples))

        self.assertEqual(len(two.source.rels), 1)
        self.assertEqual(two.nx.number_of_edges(), 2)
        self.assertEqual(two.nx.number_of_nodes(), 2)

        self.assertEqual(two.nx.nodes[0], {'label': 'custom_ent'})
        self.assertEqual(two.nx.nodes[1], {'label': '1'})

        self.assertEqual(two.nx.edges[0, 1, 0],
                         {'rid': 0, 'label': 'custom_rel'})

        self.assertEqual(two.nx.edges[1, 0, 0],
                         {'rid': 0, 'label': 'custom_rel'})

        two.str_stats

    def test_multi(self):
        """

        Test basic properties of a more complex graph.
        See definition at top to get a understanding of the
        graph properties.

        """
        m = Graph(name='multi', source=test_graph_rels)

        self.assertEqual(len(m.source.rels), 4)
        self.assertEqual(len(m.source.ents), 8)

        self.assertEqual(set(m.source.rels.keys()), set(range(4)))
        self.assertEqual(set(m.source.ents.keys()), set(range(8)))

        self.assertEqual(m.source.rels[0], 'multi')
        self.assertEqual(m.source.rels[1], 'reflexive')
        self.assertEqual(m.source.rels[2], 'symmetric')
        self.assertEqual(m.source.rels[3], 'transitive')
        self.assertEqual(m.source.ents[0], 'center')

        self.assertEqual(m.nx.number_of_edges(), 13)
        self.assertEqual(m.nx.number_of_nodes(), 8)

        m.str_stats


class SelectTests(unittest.TestCase):

    def test_edge(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(tails={2, 4}, edges={3})

        self.assertIn((0, 4, 3), sel)
        self.assertIn((4, 2, 3), sel)
        self.assertIn((0, 2, 3), sel)
        self.assertEqual(3, len(sel))

    def test_edges1(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(heads={0}, tails={2}, edges={0, 3})

        self.assertIn((0, 2, 3), sel)
        self.assertIn((0, 2, 0), sel)
        self.assertEqual(2, len(sel))

    def test_edges2(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(heads={0, 2}, tails={1, 2})

        self.assertIn((0, 1, 0), sel)
        self.assertIn((0, 2, 0), sel)
        self.assertIn((0, 2, 3), sel)
        self.assertIn((2, 2, 1), sel)

        self.assertEqual(4, len(sel))

    def test_edges_empty1(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(heads={1}, tails={1})

        self.assertEqual(0, len(sel))

    def test_edges_empty2(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(heads={6, 7}, edges={1, 2})

        self.assertEqual(0, len(sel))

    def test_combination(self):
        g = Graph(source=test_graph_rels)
        sel = g.select(
            edges={2},
            heads={0, 3, 4, 5, 6},
            tails={0, 3, 4, 5, 6})

        self.assertIn((0, 3, 2), sel)
        self.assertIn((3, 0, 2), sel)
        self.assertIn((4, 5, 2), sel)
        self.assertIn((5, 4, 2), sel)

        self.assertEqual(4, len(sel))


class FindTests(unittest.TestCase):

    @property
    def g(self) -> Graph:
        return self._g

    def setUp(self):
        self._g = Graph(source=test_graph_rels, name='original')

    def test_edge(self):
        sel = self.g.find(edges={2})

        self.assertIn((0, 3, 2), sel)
        self.assertIn((3, 0, 2), sel)
        self.assertIn((4, 5, 2), sel)
        self.assertIn((5, 4, 2), sel)
        self.assertEqual(4, len(sel))

    def test_edges(self):
        sel = self.g.find(edges={1, 2})

        self.assertIn((2, 2, 1), sel)
        self.assertIn((0, 3, 2), sel)
        self.assertIn((3, 0, 2), sel)
        self.assertIn((4, 5, 2), sel)
        self.assertIn((5, 4, 2), sel)
        self.assertEqual(5, len(sel))

    def test_head(self):
        sel = self.g.find(heads={4})

        self.assertIn((4, 5, 2), sel)
        self.assertIn((4, 2, 3), sel)
        self.assertEqual(2, len(sel), sel)

    def test_tail(self):
        sel = self.g.find(tails={4})

        self.assertIn((0, 4, 3), sel)
        self.assertIn((5, 4, 2), sel)
        self.assertEqual(2, len(sel), sel)

    def test_heads(self):
        sel = self.g.find(heads={4, 6})

        self.assertIn((4, 5, 2), sel)
        self.assertIn((4, 2, 3), sel)
        self.assertIn((6, 3, 0), sel)
        self.assertIn((6, 7, 0), sel)
        self.assertEqual(4, len(sel))

    def test_tails(self):
        sel = self.g.find(tails={4, 1})

        self.assertIn((0, 1, 0), sel)
        self.assertIn((0, 4, 3), sel)
        self.assertIn((5, 4, 2), sel)
        self.assertEqual(3, len(sel))

    def test_combination_weak(self):
        sel = self.g.find(heads={1}, tails={1}, edges={2})

        self.assertIn((0, 1, 0), sel)
        self.assertIn((0, 3, 2), sel)
        self.assertIn((3, 0, 2), sel)
        self.assertIn((4, 5, 2), sel)
        self.assertIn((5, 4, 2), sel)
        self.assertEqual(5, len(sel))


class CloneGraphTests(unittest.TestCase):

    def test_clone(self):
        g1 = Graph(name='name', source=test_graph_rels)
        g2 = g1.clone()

        self.assertIsNot(g1, g2)
        self.assertIsNot(g1.nx, g2.nx)

        self.assertEqual(g1.source, g2.source)
        self.assertEqual('name', g2.name)

        self.assertEqual(g1.nx.number_of_nodes(), g2.nx.number_of_nodes())
        self.assertEqual(g1.nx.number_of_edges(), g2.nx.number_of_edges())

        for node in g1.nx.nodes:
            self.assertIn(node, g2.nx.nodes)

        for edge in g1.nx.edges:
            self.assertIn(edge, g2.nx.edges)

    def test_clone_references(self):
        g1 = Graph(name='name', source=test_graph_rels)

        g1.nx.nodes[0]['immutable'] = 'immutable'
        g1.nx.nodes[1]['mutable'] = {'mutable': True}

        g2 = g1.clone()

        g1.nx.nodes[0]['immutable'] = 'something else'
        g2.nx.nodes[3]['new'] = 'new'

        self.assertEqual('immutable', g2.nx.nodes[0]['immutable'])
        self.assertIs(g1.nx.nodes[1]['mutable'], g2.nx.nodes[1]['mutable'])


class JoinGraphTests(unittest.TestCase):

    def test_join_empty(self):
        g1 = Graph(name='graph1')
        g2 = Graph(name='graph2')

        g1.join(g2)

        self.assertEqual(g1.name, 'graph1')
        self.assertEqual(len(g1.source.rels), 0)

        self.assertEqual(g1.source.triples, set())
        self.assertEqual(g1.source.rels, {})
        self.assertEqual(g1.source.ents, {})

        g1.str_stats

    def test_join_empty_with_one(self):
        triples = [(0, 0, 0)]

        g1 = Graph(name='graph1')
        g2 = Graph(name='graph2', source=GraphImport(triples=triples))

        g1.join(g2)

        self.assertEqual(g1.source.triples, set(triples))

        self.assertEqual(len(g1.source.rels), 1)
        self.assertEqual(g1.nx.number_of_edges(), 1)
        self.assertEqual(g1.nx.number_of_nodes(), 1)
        self.assertEqual(g1.nx.nodes[0], {'label': '0'})
        self.assertEqual(g1.nx.edges[0, 0, 0], {'rid': 0, 'label': '0'})

    def test_join_with_overwrite(self):
        s1 = GraphImport(
            triples=[(0, 1, 0), (1, 0, 0), (3, 0, 0)],
            rels={0: 'rel_nope'},
            ents={0: 'ent_nope', 1: 'ent_nope', 3: 'ent_yep'})

        s2 = GraphImport(
            triples=[(0, 1, 0), (1, 0, 1), (1, 2, 1)],
            rels={0: 'rel_yep'},
            ents={1: 'ent_yep'})

        g1 = Graph(source=s1)
        g2 = Graph(source=s2)

        self.assertEqual(g1.nx.number_of_nodes(), 3)
        self.assertEqual(g1.nx.number_of_edges(), 3)

        g1.join(g2)

        self.assertEqual(g1.nx.number_of_nodes(), 4)
        self.assertEqual(g1.nx.number_of_edges(), 5)

        self.assertIn((0, 1, 0), g1.nx.edges)
        self.assertIn((1, 0, 0), g1.nx.edges)
        self.assertIn((1, 0, 1), g1.nx.edges)
        self.assertIn((1, 2, 1), g1.nx.edges)
        self.assertIn((3, 0, 0), g1.nx.edges)

        self.assertEqual(g1.source.rels[0], 'rel_yep')
        self.assertEqual(g1.source.rels[1], '1')

        self.assertEqual(g1.source.ents[0], '0')
        self.assertEqual(g1.source.ents[1], 'ent_yep')
        self.assertEqual(g1.source.ents[2], '2')
        self.assertEqual(g1.source.ents[3], 'ent_yep')

    def test_join_no_edges(self):
        g1 = Graph()
        g1.nx.add_node(0)

        g2 = Graph()
        g2.nx.add_node(1)

        g2.join(g1)

        self.assertEqual(g2.nx.number_of_nodes(), 2)
        self.assertEqual(g2.nx.number_of_edges(), 0)

        self.assertIn(0, g2.nx.nodes)
        self.assertIn(1, g2.nx.nodes)

    def test_join_source_equal(self):
        g1 = Graph(source=test_graph_rels)
        g2 = Graph()

        g2.join(g1)
        self.assertEqual(g1.source, g2.source)


class SubGraphTests(unittest.TestCase):

    def test_from_one(self):
        g = Graph(source=GraphImport(
            triples=[(0, 0, 0)],
            ents={0: 'zero'}),
        )

        nid, sub = g.sample_subgraph(start=0, depth=3)

        self.assertEqual(nid, 0)

        self.assertEqual(sub.nx.number_of_edges(), 1)
        self.assertEqual(sub.nx.number_of_nodes(), 1)

        self.assertEqual(sub.nx.edges[0, 0, 0], {'rid': 0, 'label': '0'})
        self.assertEqual(sub.nx.nodes[0], {'label': 'zero'})

    def test_to_one(self):
        g = Graph(source=test_graph_rels)
        nid, sub = g.sample_subgraph(start=2, depth=0)

        self.assertEqual(sub.nx.number_of_nodes(), 1)
        self.assertEqual(sub.nx.number_of_edges(), 0)

    def test_small(self):
        g = Graph(source=test_graph_rels)
        nid, sub = g.sample_subgraph(start=6, depth=1)

        self.assertEqual(sub.nx.number_of_nodes(), 3)
        self.assertEqual(sub.nx.number_of_edges(), 2)

        self.assertIn((6, 7, 0), sub.nx.edges)
        self.assertIn((6, 3, 0), sub.nx.edges)

        self.assertIs(sub.source, g.source)

    def test_deeper(self):
        g = Graph(source=test_graph_rels)
        nid, sub = g.sample_subgraph(start=5, depth=2)

        self.assertEqual(nid, 5)

        self.assertEqual(sub.nx.number_of_nodes(), 6)
        self.assertEqual(sub.nx.number_of_edges(), 6)

        self.assertIn((5, 6, 0), sub.nx.edges)
        self.assertIn((6, 7, 0), sub.nx.edges)
        self.assertIn((6, 3, 0), sub.nx.edges)
        self.assertIn((5, 4, 2), sub.nx.edges)
        self.assertIn((4, 5, 2), sub.nx.edges)
        self.assertIn((4, 2, 3), sub.nx.edges)

    def test_outermost_symmetric(self):
        g = Graph(source=test_graph_rels)
        nid, sub = g.sample_subgraph(start=0, depth=2)

        self.assertIn((4, 5, 2), sub.nx.edges)
        self.assertIn((5, 4, 2), sub.nx.edges)
        self.assertIn((2, 2, 1), sub.nx.edges)

        self.assertEqual(sub.nx.number_of_nodes(), 6)
        self.assertEqual(sub.nx.number_of_edges(), 10)


class PathFindTests(unittest.TestCase):

    def test_empty_path(self):
        pass  # 0, 7 cutoff=2

    def test_depth0(self):
        g = Graph(source=test_graph_paths)

        paths = {
            path.rels: path
            for path in g.find_paths(2, 5, cutoff=5, depth=0)
        }

        self.assertEqual(len(paths), 3)

        # 1-hop

        self.assertIn((0, ), paths)
        path = paths[(0, )]

        self.assertEqual(len(path), 1)
        self.assertEqual((2, 5), path.ents)
        self.assertEqual((0, ), path.rels)

        self.assertEqual('head', path.g.nx.nodes[2]['kind'])
        self.assertEqual('tail', path.g.nx.nodes[5]['kind'])
        self.assertEqual('path', path.g.nx.edges[2, 5, 0]['kind'])

        for node in (set(test_graph_paths.ents.keys()) - set(path.ents)):
            with self.assertRaises(KeyError):
                path.g.nx.nodes[node]['kind']

        # 2-hop

        for key in ((0, 0), (1, 0)):
            self.assertIn(key, paths)
            path = paths[key]

            self.assertEqual(len(path), 2)
            self.assertEqual((2, 4, 5), path.ents)
            self.assertEqual(path.g.nx.number_of_nodes(), 3)
            self.assertEqual(path.g.nx.number_of_edges(), 5)

            self.assertIn((2, 4, 0), path.g.nx.edges)
            self.assertIn((2, 4, 1), path.g.nx.edges)
            self.assertIn((4, 5, 0), path.g.nx.edges)
            self.assertIn((5, 4, 0), path.g.nx.edges)
            self.assertIn((2, 5, 0), path.g.nx.edges)

            self.assertEqual(g.source, path.g.source)

            self.assertEqual('head', path.g.nx.nodes[path.ents[0]]['kind'])
            self.assertEqual('tail', path.g.nx.nodes[path.ents[-1]]['kind'])
            for node in path.ents[1:-1]:
                self.assertEqual('path', path.g.nx.nodes[node]['kind'])

            # FIXME test path annotation on edges

            for node in (set(test_graph_paths.ents.keys()) - set(path.ents)):
                with self.assertRaises(KeyError):
                    path.g.nx.nodes[node]['kind']

    def test_depth1(self):
        g = Graph(source=test_graph_paths)

        paths = {
            path.rels: path
            for path in g.find_paths(2, 5, cutoff=5, depth=1)
        }

        for path in paths.values():
            self.assertEqual(6, path.g.nx.number_of_nodes())
            self.assertNotIn(0, path.g.nx.nodes)

        self.assertEqual(9, paths[(0, )].g.nx.number_of_edges())
        self.assertNotIn((4, 4, 0), paths[(0, )].g.nx.edges)

        self.assertEqual(10, paths[(0, 0)].g.nx.number_of_edges())
        self.assertEqual(10, paths[(1, 0)].g.nx.number_of_edges())
        self.assertIn((4, 4, 0), paths[(0, 0)].g.nx.edges)
        self.assertIn((4, 4, 0), paths[(1, 0)].g.nx.edges)

    def test_cutoff(self):
        g = Graph(source=test_graph_paths)
        paths = list(g.find_paths(0, 2, cutoff=1))
        self.assertEquals(0, len(paths))


class StrollTests(unittest.TestCase):

    def _check_grounds(self, stroll, pattern, grounds):
        ref = stroll.grounds[pattern]

        for walk in grounds:
            self.assertIn(walk, ref, ref)

        self.assertEqual(len(grounds), len(ref), ref)

    def test_zero(self):
        g = Graph(source=test_graph_paths)
        stroll = g.stroll(center=2, cutoff=0)

        self.assertEqual(0, len(stroll.walks))
        self.assertEqual(0, len(stroll.patterns))
        self.assertEqual(0, len(stroll.grounds))

    def test_small(self):
        g = Graph(source=test_graph_paths)
        stroll = g.stroll(center=2, cutoff=1)

        for walk in stroll.walks:
            self.assertEqual(2, len(walk), str(walk))

        for pattern in stroll.patterns:
            self.assertEqual(1, len(pattern), str(pattern))

        # walks

        walks = (2, 3), (2, 1), (2, 4), (2, 5)
        for walk in walks:
            self.assertIn(walk, stroll.walks)

        self.assertEqual(len(walks), len(stroll.walks))

        # patterns

        self.assertIn((1, ), stroll.patterns)
        self.assertIn((0, ), stroll.patterns)
        self.assertEqual(2, len(stroll.patterns))

        # grounds

        self.assertIn((2, 4), stroll.grounds[(1, )])

        for walk in walks:
            self.assertIn(walk, stroll.grounds[(0, )])

    def test_big(self):
        g = Graph(source=test_graph_paths)
        stroll = g.stroll(center=2, cutoff=2)

        walks = ((2, 3),
                 (2, 1),
                 (2, 1, 2),
                 (2, 1, 0),
                 (2, 5, 6),
                 (2, 5), (2, 4, 5),
                 (2, 4), (2, 4, 4), (2, 5, 4))

        patterns = ((0, ), (0, 0), (1, ), (1, 0))

        # walks

        for walk in walks:
            self.assertIn(walk, stroll.walks)
        self.assertEqual(len(walks), len(stroll.walks))

        # patterns

        for pattern in patterns:
            self.assertIn(pattern, stroll.patterns)
        self.assertEqual(len(patterns), len(stroll.patterns))

        # grounds

        pattern = (0, )
        grounds = (2, 3), (2, 1), (2, 4), (2, 5)
        self._check_grounds(stroll, pattern, grounds)

        pattern = (0, 0)
        grounds = (
            (2, 1, 2), (2, 1, 0), (2, 5, 6),
            (2, 4, 5), (2, 4, 4), (2, 5, 4))
        self._check_grounds(stroll, pattern, grounds)

        pattern = (1, )
        grounds = ((2, 4), )
        self._check_grounds(stroll, pattern, grounds)

        pattern = (1, 0)
        grounds = ((2, 4, 4), (2, 4, 5), )
        self._check_grounds(stroll, pattern, grounds)

    def test_cycle_small(self):
        source_cycle_small = GraphImport(triples=[
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
        ])

        g = Graph(source=source_cycle_small)
        stroll = g.stroll(center=0, cutoff=10)

        # not (0, 1, 0, 1, ...), (0, 1, 1, 1, ...)
        self.assertIn((0, 1), stroll.walks)
        self.assertIn((0, 1, 0), stroll.walks)
        self.assertEqual(3, len(stroll.walks), stroll.walks)

        self.assertIn((0, ), stroll.patterns)
        self.assertIn((0, 0), stroll.patterns)
        self.assertEqual(2, len(stroll.patterns))

    def test_cycle_big(self):
        source_cycle_big = GraphImport(triples=[
            (0, 1, 0),
            (1, 1, 0),
            (1, 2, 0),
            (2, 2, 0),
            (2, 0, 0),
        ])

        g = Graph(source=source_cycle_big)
        stroll = g.stroll(center=0, cutoff=100)

        walks = (
            # debatable: should there be (0, 1, 2, 0, 1)?
            (0, 1), (0, 1, 1),
            # debatable: should there be
            # (0, 1, 1, 2),
            # (0, 1, 1, 2, 2)?
            (0, 1, 2), (0, 1, 2, 2),
            # debatable: should there be
            # (0, 1, 1, 2, 0),
            # (0, 1, 2, 2, 0),
            # (0, 1, 1, 2, 2, 0) ?
            (0, 1, 2, 0))

        for walk in walks:
            self.assertIn(walk, stroll.walks)
        self.assertEqual(len(walks), len(stroll.walks), stroll.walks)

    def test_to(self):
        g = Graph(source=test_graph_paths)
        stroll = g.stroll(center=2, cutoff=2).to(4)

        def check(pattern, walks):
            self.assertIn(pattern, stroll.grounds)

            for walk in walks:
                self.assertIn(walk, stroll.grounds[pattern])

            self.assertEqual(len(walks), len(stroll.grounds[pattern]))

        # grounds

        pattern = (0, )
        walks = ((2, 4), )
        check(pattern, walks)

        pattern = (0, 0)
        walks = (2, 4, 4), (2, 5, 4)
        check(pattern, walks)

        pattern = (1, )
        walks = ((2, 4), )
        check(pattern, walks)

        pattern = (1, 0)
        walks = ((2, 4, 4), )
        check(pattern, walks)

    def test_to_self(self):
        g = Graph(source=test_graph_rels)
        stroll = g.stroll(center=0, cutoff=5).to(0)

        patterns = (2, 2), (3, 2, 0, 0, 2)
        walks = (0, 3, 0), (0, 4, 5, 6, 3, 0)

        for pattern, walk in zip(patterns, walks):
            self.assertIn(pattern, stroll.grounds)
            self.assertIn(walk, stroll.grounds[pattern])
            self.assertEqual(1, len(stroll.grounds[pattern]))

        self.assertEqual(len(patterns), len(stroll.grounds))


class ConnectingEdgesTests(unittest.TestCase):

    def test_connecting_edges(self):
        g = Graph(source=test_graph_rels)
        trail = 3, 0, 2, 2
        pattern = set(g.connecting_edges(trail=trail))

        self.assertIn((2, 0, 1), pattern)
        self.assertIn((2, 3, 1), pattern)
        self.assertEqual(2, len(pattern))


class PartitionTests(unittest.TestCase):

    @property
    def g(self) -> Graph:
        return self._g

    def setUp(self):
        self._g = Graph(source=test_graph_rels, name='original')

    def test_empty_partition(self):
        with self.assertRaises(PartitionConstraintError):
            self.g.partition()

    def test_too_small(self):
        with self.assertRaises(PartitionConstraintError):
            self.g.partition(p1={1}, p2={0, 2, 3, 4, 5, 6, 7})

    def test_incomplete_sets(self):
        with self.assertRaises(PartitionConstraintError):
            self.g.partition(p1={0, 2, 3})

    def test_nondisjunctive(self):
        with self.assertRaises(PartitionConstraintError):
            self.g.partition(p1={0, 1, 2, 3}, p2={3, 4, 5, 6, 7})

    def test_unconnected(self):
        with self.assertRaises(PartitionConstraintError):
            self.g.partition(p1={1, 4}, p2={0, 2, 3, 5, 6, 7})

    def test_partition1(self):
        parts = self.g.partition(p1={0, 1, 2}, p2={3, 4, 5, 6, 7})

        self.assertIn('p1', parts)
        self.assertIn('p2', parts)
        self.assertEqual(len(parts), 2)

        gp1, gp2 = parts['p1'], parts['p2']

        # gp1

        self.assertEqual(gp1.nx.number_of_edges(), 4)
        self.assertEqual(gp1.nx.number_of_nodes(), 3)

        self.assertIn((0, 1, 0), gp1.nx.edges)
        self.assertIn((0, 2, 0), gp1.nx.edges)
        self.assertIn((0, 2, 3), gp1.nx.edges)
        self.assertIn((2, 2, 1), gp1.nx.edges)

        self.assertNotIn((4, 5, 2), gp1.nx.edges)
        self.assertNotIn((5, 4, 2), gp1.nx.edges)
        self.assertNotIn((5, 6, 0), gp1.nx.edges)
        self.assertNotIn((6, 7, 0), gp1.nx.edges)

        # gp2

        self.assertEqual(gp2.nx.number_of_edges(), 5)
        self.assertEqual(gp2.nx.number_of_nodes(), 5)

        self.assertNotIn((0, 1, 0), gp2.nx.edges)
        self.assertNotIn((0, 2, 0), gp2.nx.edges)
        self.assertNotIn((0, 2, 3), gp2.nx.edges)
        self.assertNotIn((2, 2, 1), gp2.nx.edges)

        self.assertIn((4, 5, 2), gp2.nx.edges)
        self.assertIn((5, 4, 2), gp2.nx.edges)
        self.assertIn((5, 6, 0), gp2.nx.edges)
        self.assertIn((6, 7, 0), gp2.nx.edges)

        # both

        self.assertNotIn((0, 3, 2), gp1.nx.edges)
        self.assertNotIn((0, 3, 2), gp2.nx.edges)

        self.assertNotIn((3, 0, 2), gp1.nx.edges)
        self.assertNotIn((3, 0, 2), gp2.nx.edges)

        self.assertNotIn((0, 4, 3), gp1.nx.edges)
        self.assertNotIn((0, 4, 3), gp2.nx.edges)

    def test_partition2(self):
        parts = self.g.partition(p1={0, 1, 3}, p2={2, 4}, p3={5, 6, 7})

        self.assertIn('p1', parts)
        self.assertIn('p2', parts)
        self.assertIn('p3', parts)
        self.assertEqual(len(parts), 3)

        gp1, gp2, gp3 = parts['p1'], parts['p2'], parts['p3']

        # gp1

        self.assertEqual(gp1.nx.number_of_edges(), 3)
        self.assertEqual(gp1.nx.number_of_nodes(), 3)
        self.assertIn((0, 1, 0), gp1.nx.edges)
        self.assertIn((0, 3, 2), gp1.nx.edges)
        self.assertIn((3, 0, 2), gp1.nx.edges)

        # gp2

        self.assertEqual(gp2.nx.number_of_edges(), 2)
        self.assertEqual(gp2.nx.number_of_nodes(), 2)
        self.assertIn((4, 2, 3), gp2.nx.edges)
        self.assertIn((2, 2, 1), gp2.nx.edges)

        # gp2

        self.assertEqual(gp3.nx.number_of_edges(), 2)
        self.assertEqual(gp3.nx.number_of_nodes(), 3)
        self.assertIn((5, 6, 0), gp3.nx.edges)
        self.assertIn((6, 7, 0), gp3.nx.edges)

    def test_partition_names(self):
        parts = self.g.partition(p1={0, 1, 2}, p2={3, 4, 5, 6, 7})
        gp1, gp2 = parts['p1'], parts['p2']

        self.assertEqual(gp1.name, 'original-p1')
        self.assertEqual(gp2.name, 'original-p2')

    def test_partition_names_prefixed(self):
        parts = self.g.partition(
            prefix='part',
            p1={0, 1, 2},
            p2={3, 4, 5, 6, 7})

        gp1, gp2 = parts['p1'], parts['p2']

        self.assertEqual(gp1.name, 'part-p1')
        self.assertEqual(gp2.name, 'part-p2')

    def test_partition_edge_data(self):
        # this deletes all relations of type 2
        parts = self.g.partition(p1={0, 1, 2, 4}, p2={3, 5, 6, 7})

        gp1, gp2 = parts['p1'], parts['p2']

        self.assertEqual(
            gp1.nx.edges[0, 1, 0],
            {'rid': 0, 'label': test_graph_rels.rels[0]})

        self.assertEqual(
            gp1.nx.edges[0, 2, 0],
            {'rid': 0, 'label': test_graph_rels.rels[0]})

        self.assertEqual(
            gp1.nx.edges[0, 2, 3],
            {'rid': 3, 'label': test_graph_rels.rels[3]})

        self.assertEqual(
            gp1.nx.edges[2, 2, 1],
            {'rid': 1, 'label': test_graph_rels.rels[1]})

        self.assertEqual(
            gp1.nx.edges[4, 2, 3],
            {'rid': 3, 'label': test_graph_rels.rels[3]})

        self.assertIn(0, gp1.source.rels)
        self.assertNotIn(2, gp1.source.rels)
        self.assertIn(1, gp1.source.rels)
        self.assertIn(3, gp1.source.rels)

        # --

        self.assertEqual(
            gp2.nx.edges[5, 6, 0],
            {'rid': 0, 'label': test_graph_rels.rels[0]})

        self.assertEqual(
            gp2.nx.edges[6, 3, 0],
            {'rid': 0, 'label': test_graph_rels.rels[0]})

        self.assertEqual(
            gp2.nx.edges[6, 7, 0],
            {'rid': 0, 'label': test_graph_rels.rels[0]})

        self.assertIn(0, gp2.source.rels)
        self.assertNotIn(2, gp2.source.rels)
        self.assertNotIn(1, gp2.source.rels)
        self.assertNotIn(3, gp2.source.rels)


class InductionTests(unittest.TestCase):

    @property
    def g(self) -> Graph:
        return self._g

    def setUp(self):
        self._g = Graph(source=test_graph_rels, name='original')

    def test_empty_induction(self):
        subg = self.g.induce(triples=set())

        self.assertEqual(0, len(subg.nx.edges))
        self.assertEqual(0, len(subg.source.rels))
        self.assertEqual(0, len(subg.source.ents))
        self.assertEqual(0, len(subg.source.triples))

    def test_find_induction(self):
        ents = {0, 1, 2}
        sub = self.g.induce(triples=self.g.select(heads=ents, tails=ents))

        self.assertEqual(4, len(sub.nx.edges))
        self.assertEqual(3, len(sub.nx.nodes))

        self.assertIn(0, sub.nx.nodes)
        self.assertIn(1, sub.nx.nodes)
        self.assertIn(2, sub.nx.nodes)

        self.assertIn((0, 1, 0), sub.nx.edges)
        self.assertIn((0, 2, 0), sub.nx.edges)
        self.assertIn((0, 2, 3), sub.nx.edges)
        self.assertIn((2, 2, 1), sub.nx.edges)

        self.assertEqual(3, len(sub.source.rels))  # not 2
        self.assertEqual(3, len(sub.source.ents))
        self.assertEqual(4, len(sub.source.triples))

    def test_name_induction(self):
        name = 'foo'
        subg = self.g.induce(triples=set(), name=name)
        self.assertEqual(subg.name, name)
