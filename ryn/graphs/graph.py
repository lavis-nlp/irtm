# -*- coding: utf-8 -*-

"""

ryn specific graph abstractions and utilities

"""

from ryn.common import logging

import networkx
import numpy as np

import queue
import pickle
import pathlib

from dataclasses import field
from dataclasses import dataclass
from dataclasses import FrozenInstanceError
from collections import Counter
from collections import defaultdict


from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import NewType


log = logging.get('graphs.graph')


Triple = NewType('Triple', Tuple[int, int, int])


class PartitionConstraintError(Exception):
    """
    Thrown for violated constraints in Graph.partition.
    """
    pass


# ----------| UTILITY


def condense(lis: List[Tuple[int, int, int]]) -> List[List[int]]:
    """

    Condenses list of triples as if the first tuple was a key.

    Example
    -------

    Given: [
      [0, 1, 10],
      [0, 1, 20],
      [1, 2, 40], ]

    Returns: [[10, 20], [40]]

    """
    a, b = None, None
    res = []

    for x, y, z in lis:
        if a == x and b == y:
            res[-1].append(z)
        else:
            res.append([z])
            a, b = x, y

    return res


def permutations(lis: List[List[Any]]):
    """

    Takes a list of lists of options and returns
    all possible decisions:

    Example
    -------

    Given: lis: [[10, 20], [30], [40, 50]]
    Returns: [
      [10, 30, 40],
      [20, 30, 40],
      [10, 30, 50],
      [20, 30, 50],]

    """
    res = [[s] for s in lis[0]]

    for options in lis[1:]:
        buf = []

        for option in options:
            for path in res:
                buf.append(path + [option])

        res = buf

    # should not be necessary...
    # I am paranoid now
    for perm in res:
        assert len(perm) == len(lis)

    return tuple(res)


def connecting_edges(g_nx, nodes: Tuple[int] = None):
    """

    Get all edge identifier based on the list of nodes.

    Parameters
    ----------

    g_nx : networkx.DiMultiGraph
      Graph to sample from

    nodes : Tuple[int]
      Sequence of nodes to be visited

    """
    assert False, 'deprecated!'

    assert nodes is not None
    assert type(nodes) is tuple, type(nodes)

    # build a set of tuples forming the path
    # each tups[i] = (int, int): (h, t)
    tups = set(zip(nodes, nodes[1:]))

    # retrieve all neighbours for each node on the path
    # each cand[i]: (int, int, int) = (h, t, r)
    edges = g_nx.edges(nodes, data=True)
    trips = map(lambda t: (t[0], t[1], t[2]['rid']), edges)

    # only consider those strictly following the path
    # (i.e. no transitive shortcuts, no wrong direction)
    # and condense this into a list of collections of edge ids
    edge_agg = filter(lambda e: (e[0], e[1]) in tups, trips)
    edge_agg = condense(edge_agg)

    # create all possible paths
    yield from map(tuple, permutations(edge_agg))


# ----------| DATA


class frozendict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._frozen = True

    def __setitem__(self, *args, **kwargs):
        try:
            self._frozen

        except AttributeError:
            super().__setitem__(*args, **kwargs)
            return

        raise FrozenInstanceError('mapping is frozen')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._frozen = True


@dataclass(frozen=True)
class GraphImport:
    """

    Unified data definition used by ryn.graph.Graph.

    Graph triples are of the following structure: (head, tail, relation)
    You can provide any Iterable for the triples. They are converted
    to frozenset[Tuple[int, int, int]]

    Currently the graph is defined by it's edges, which means each
    node is at least connected to one other node. This might change in
    the future.

    Order of provided triples is not preserved.

    The rels and ents dictionaries are filled with all missing
    information automatically such that e.g. rels[i] = f'{i}'.
    They cannot be changed afterwards.

    """

    # (head, tail, relation)
    triples: Set[Tuple[int, int, int]]

    rels: Dict[int, str] = field(default_factory=dict)
    ents: Dict[int, str] = field(default_factory=dict)

    # --

    # not using _set_all for backwards compatibility
    # with older pickled versions - FIXME migrate?
    @property
    def e2id(self):
        try:
            self._e2id
        except AttributeError:
            rev = frozendict({v: k for k, v in self.ents.items()})
            self._set('_e2id', rev)

        return self._e2id

    @property
    def r2id(self):
        try:
            self._r2id
        except AttributeError:
            rev = frozendict({v: k for k, v in self.ents.items()})
            self._set('_r2id', rev)

        return self._r2id

    # ---

    def _set(self, prop: str, *args, **kwargs):
        object.__setattr__(self, prop, *args, **kwargs)

    def _set_all(self, triples, ents, rels):
        self._set('triples', frozenset(triples))
        self._set('ents', frozendict(ents))
        self._set('rels', frozendict(rels))

    def _resolve(self, idx: int, d: Dict[int, str]):
        if idx not in d:
            label = str(idx)
            d[idx] = label

    def __post_init__(self):
        triples = set(map(tuple, self.triples))

        for h, t, r in self.triples:
            self._resolve(h, self.ents)
            self._resolve(t, self.ents)
            self._resolve(r, self.rels)

        self._set_all(triples, self.ents, self.rels)

    # --

    def join(self, other: 'GraphImport') -> 'GraphImport':
        ents = {**self.ents, **other.ents}
        rels = {**self.rels, **other.rels}
        triples = self.triples | other.triples

        self._set_all(triples, ents, rels)

    # ---

    @classmethod
    def induce(C, g: 'Graph', triples: Set[Triple]):
        if len(triples):
            heads, tails, rels = map(set, zip(*triples))
        else:
            heads, tails, rels = set(), set(), set()

        return C(
            triples=triples,
            rels={
                k: v for k, v in g.source.rels.items()
                if k in rels},
            ents={
                k: v for k, v in g.source.ents.items()
                if k in heads | tails}, )


@dataclass
class Path:
    """

    This class is used to model paths through the multi-edge, directed
    graphs used here. The neighbouring nodes of the paths nodes can
    also be supplied and the parameter denoting the subgraph's depth
    is given.

    The rels list is always len(nodes) - 1. ents[i] is connected by
    rels[i] with ents[i + 1].

    """

    depth: int
    g: 'Graph'

    ents: Tuple[int]  # sequence of connected nodes
    rels: Tuple[int]  # sequence of edge kinds

    def __len__(self):
        return len(self.rels)

    def __str__(self):
        zipped = zip(self.ents, self.rels)
        s = '-'.join([f'({t[0]})-[{t[1]}]' for t in zipped])
        return s + f'-({self.ents[-1]})'


@dataclass
class WalkAggregator:

    walk:    List[int]
    pattern: List[int]

    visited: Counter = field(default_factory=Counter)

    @property
    def packed(self):
        return (self.walk, self.pattern, dict(self.visited))

    @staticmethod
    def unpack(data):
        walk, pattern, visited = data
        return WalkAggregator(
            walk=walk,
            pattern=pattern,
            visited=Counter(visited))


Grounds = Dict[Tuple[int], Set[Tuple[int]]]


@dataclass
class Stroll:
    """

    See Graph.stroll and path.Index

    basic assumption (memory-wise): len(patterns) << len(walks)

    """
    # pattern -> walks
    grounds: Grounds = field(
        default_factory=lambda: defaultdict(set))

    @property
    def walks(self) -> Set[Tuple[int]]:
        if not len(self.grounds):
            return set()

        return set.union(*self.grounds.values())

    @property
    def patterns(self) -> Set[Tuple[int]]:
        return set(self.grounds.keys())

    def add(self, agg: WalkAggregator):
        for i in range(len(agg.pattern)):
            pattern = tuple(agg.pattern[:i+1])
            walk = tuple(agg.walk[:i+2])
            self.grounds[pattern].add(walk)

    def to(self, t: int) -> 'Stroll':
        stroll = Stroll()

        for pattern, walks in self.grounds.items():
            for walk in walks:
                if walk[-1] == t:
                    stroll.grounds[pattern].add(walk)

        return stroll

    # -- persistence

    @property
    def packed(self):
        return tuple(
            (pattern, tuple(walks))
            for pattern, walks in self.grounds.items()
        )

    @staticmethod
    def unpack(data):
        stroll = Stroll()

        for pattern, walks in data:
            walks = (tuple(w) for w in walks)
            stroll.grounds[tuple(pattern)] |= set(walks)

        return stroll


# ----------| NETWORKX


class Graph:
    """
    Ryn Graph Implementation

    Create a new graph object which maintains a networkx graph.
    This class serves as a provider of utilities working on
    and for initializing the networkx graph.

    Design Decisions
    ----------------

    Naming:

    Naming nodes and edges: networkx uses "nodes" and "edges". To
    not confuse on which "level" you operate on the graph, everything
    here is called "ents" (for entities) and "rels" (for relations)
    when working with Ryn code and "node" and "edges" when working
    with networkx instances.

    Separate Relation and Entitiy -Mapping:

    The reasoning for not providing (e.g.) the Graph.source.rels
    mapping directly on the graph is to avoid a false expectation
    that this is automatically in sync with the graph itself.
    Consider manipulating Graph.g (deleting nodes for example) -
    this would not update the .rels-mapping. Thus this is explicitly
    separated in the .source GraphImport.

    """

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> str:
        self._name = val

    @property
    def source(self) -> GraphImport:
        return self._source

    @property
    def nx(self) -> networkx.MultiDiGraph:
        return self._nx

    @property
    def rnx(self) -> networkx.MultiDiGraph:
        return self._rnx

    @property
    def edges(self) -> Dict[int, Tuple[int]]:
        return self._edges

    @property
    def str_stats(self) -> str:
        s = (
            f'ryn graph: {self.name}\n'
            f'  nodes: {self.nx.number_of_nodes()}\n'
            f'  edges: {self.nx.number_of_edges()}'
            f' ({len(self.source.rels)} types)\n'
        )

        # statistical values

        try:
            degrees = np.array(list(self.nx.degree()))[:, 1]
            s += (
                f'  degree:\n'
                f'    mean {np.mean(degrees):.2f}\n'
                f'    median {int(np.median(degrees)):d}\n'
            )
        except IndexError:
            s += '  cannot measure degree\n'

        return s

    # --

    def __str__(self) -> str:
        return f'ryn.graph: [{self.name}] ({len(self.source.ents)} entities)'

    def __init__(self,
                 name: str = None,
                 source: GraphImport = None,
                 freeze: bool = False):

        assert type(name) is str if name is not None else True, f'{name=}'

        # properties
        self._nx = networkx.MultiDiGraph()
        self._edges = defaultdict(set)

        self._name = 'unknown' if name is None else name

        # GraphImport
        if source is not None:
            self._source = source
            self.add(source)
        else:
            self._source = GraphImport(triples=[])

        log.debug(f'created graph: \n{self.str_stats}\n')

        if freeze:
            log.debug('freezing graph')
            networkx.freeze(self.nx)

    # --

    def select(
            self,
            heads: Set[int] = None,
            tails: Set[int] = None,
            edges: Set[int] = None, ):
        """

        Select edges from the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------

        heads : Set[int]
          consider the provided head nodes

        tails : Set[int]
          consider the provided head nodes

        edges : Set[int]
          consider the provided edge classes


        Returns
        -------

        A set of edges adhering to the provided constraints.


        Notes
        -----

        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, tails, edges, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    if tails and t not in tails:
                        continue

                    for r in rs:
                        if edges and r not in edges:
                            continue

                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads, tails, edges))
        rng = set(_gen(self.rnx, tails, heads, edges, rev=True))

        return dom | rng

    # --

    def find(
            self,
            heads: Set[int] = None,
            tails: Set[int] = None,
            edges: Set[int] = None, ) -> Set[Triple]:
        """
        Find edges from the graph.

        An edge is a triple (h, t, r) and the selection is either
        the union or intersection of all edges containing one of the
        provided nodes and edges.

        The difference between Graph.find and Graph.select is that
        .find will select any edge containing any of the provided
        heads (union) or tails and .select will only choose those
        where their any combination of all provided entities occurs
        (intersection).

        Parameters
        ----------

        heads : Set[int]
          consider the provided head nodes

        tails : Set[int]
          consider the provided head nodes

        edges : Set[int]
          consider the provided edge classes

        Returns
        -------

        A set of edges adhering to the provided constraints.


        Notes
        -----

        Not using nx.subgraph as it would contain undesired edges
        (because nx.subgraph only works on node-level)

        """
        heads = set() if heads is None else heads
        tails = set() if tails is None else tails
        edges = set() if edges is None else edges

        def _gen(nxg, heads, rev=False):
            for h in heads:
                if h not in nxg:
                    continue

                for t, rs in nxg[h].items():
                    for r in rs:
                        yield (h, t, r) if not rev else (t, h, r)

        dom = set(_gen(self.nx, heads))
        rng = set(_gen(self.rnx, tails, rev=True))
        rel = {(h, t, r) for r in edges or [] for h, t in self.edges[r]}

        return dom | rng | rel

    # --

    def partition(
            self,
            prefix: str = None,
            **ents: Set[int]) -> Dict[str, 'Graph']:
        """

        Partition graph into a disjoint union of graphs.

        The provided entity sets define the partition. Relations on
        the partition border are removed. The new graphs have new
        GraphImports reflecting the new data situation.

        The following constraints apply for 'ents':

        1. The intersection of all ents sets is the empty set
        2. The union of all ents sets equals the ents of the source graph
        3. The smallest possible partition are two triples of a four-node graph

        Parameters
        ----------

        prefix : str
          The returned graphs are named "(prefix or self.name)-<kwarg>"

        **kwargs : Set[pint]
          The entities defining the partition

        Returns
        -------

        A dictionary containing the new graphs named by the keyword arguments

        """
        if len(ents) < 2:
            raise PartitionConstraintError(
                'provide at least two named entity sets')
        if any(len(e) < 2 for e in ents.values()):
            raise PartitionConstraintError(
                'provided node subset too small')
        if len(set.intersection(*ents.values())):
            raise PartitionConstraintError(
                'provided node sets are not disjunctive')
        if set.union(*ents.values()) != set(self.nx.nodes):
            raise PartitionConstraintError(
                'provided nodes are incomplete')

        ret = {}
        for name, part_ents in ents.items():
            triples = self.select(heads=part_ents, tails=part_ents)
            if not triples:
                raise PartitionConstraintError('subgraph has lone nodes')

            part_source = GraphImport.induce(self, triples)

            ret[name] = Graph(
                name=f'{(prefix or self.name)}-{name}',
                source=part_source,
                freeze=networkx.is_frozen(self.nx),
            )

        return ret

    # --

    def induce(self, triples: Set[Triple], name: str = None) -> 'Graph':
        """

        Induce new graphs based on the provided triples.

        This is a function to obtain new Graph instances based on
        triple sets. You would usually use the result of a
        Graph.find or Graph.select operation to feed this method.


        """
        return Graph(
            name=name or f'{self.name}-induction',
            source=GraphImport.induce(self, triples),
            freeze=networkx.is_frozen(self.nx))

    # --

    def connecting_edges(self, trail: Tuple[int] = None):
        """

        """
        assert trail is not None

        acc = []
        for h, t in zip(trail, trail[1:]):
            acc.append([r for _, _, r in self.select(heads={h}, tails={t})])

        yield from map(tuple, permutations(acc))

    # --

    def _safe_add(self,
                  target: 'Graph',
                  source: 'Graph',
                  edge):

        h, t, data = edge
        r = data['rid']

        try:
            # Check whether the relation already exists. Adding new
            # edges to a MultiDiGraph is not idempotent even with
            # equal data dicts
            target.nx.edges[h, t, r]

        except KeyError:
            target.nx.add_edge(h, t, r, **source.nx.edges[h, t, r])
            target.nx.add_node(t, **source.nx.nodes[t])  # idempotent

    def _bf_safe_add(self,
                     subg: 'Graph',
                     edge,
                     outermost: bool = False):

        h, t, _ = edge

        # This adds all symmetrical relations for the very last
        # breadth. Currently the bfs runs for one loop further than
        # necessary but only adds edges pointing to nodes already in
        # subg. Also ignore reflexive relations as they would
        # introduce an additional hop.
        if outermost:
            if t not in subg.nx or h == t:
                return

        self._safe_add(subg, self, edge)

    def _bf_create(self, nid: int, depth: int) -> 'Graph':
        subg = Graph()
        subg.nx.add_node(nid, **self.nx.nodes[nid])

        q = queue.Queue()
        q.put((1, nid))

        while not q.empty():
            d, nid = q.get()
            edges = self.nx.edges([nid], data=True)

            for edge in edges:
                _, t, _ = edge

                # also d==depth: see _bf_safe_add - outermost
                if d <= depth and t not in subg.nx:
                    q.put((d + 1, t))

                outermost = d == (depth + 1)
                self._bf_safe_add(subg, edge, outermost=outermost)

        return subg

    def sample_subgraph(self,
                        depth: int = None,
                        start: int = None) -> (
                            Tuple[int, 'Graph']):
        """
        Sample a new graph using a breadth-first approach.
        The new graph will not have a .source.

        Parameters
        ----------

        depth : int
          At most hops away from the center node

        start : int
          id of the center node, sampled randomly if None

        Returns
        -------
        subgraph: Graph
          A subgraph with all nodes at most "depth" hops away from "start"


        """
        assert depth is not None

        N = self.nx.number_of_nodes()
        nid = np.random.randint(0, N) if start is None else start

        log.debug(f'sampling subgraph of depth {depth}')
        log.debug(f'selecting node {nid} of {N} nodes')

        subg = self._bf_create(nid, depth)
        subg._source = self.source

        if networkx.is_frozen(self.nx):
            networkx.freeze(subg.nx)

        return nid, subg

    # ---

    def _annotate_path(self,
                       ents: Tuple[int],
                       rels: Tuple[int]) -> (
                           'Graph'):

        k = 'kind'

        self.nx.nodes[ents[0]][k] = 'head'
        self.nx.nodes[ents[-1]][k] = 'tail'
        for ent in ents[1:-1]:
            self.nx.nodes[ent][k] = 'path'

        #   ents          rels              edges
        # [0, 1, 2] and [10, 20] -> [(0, 1, 10), (1, 2, 20)]
        for edge in zip(ents, ents[1:], rels):
            self.nx.edges[edge][k] = 'path'

        return self

    def find_paths(self,
                   h: int = None,
                   t: int = None,
                   depth: int = 0,
                   cutoff: int = 5) -> (
                       Path):
        """

        Returns all paths connecting h with t as subgraphs.
        See also the docs for networkx's simple_paths().

        Note that each path is a complete subgraph of the source graph.
        This means, that if two nodes are connected by multiple relations,
        for each relation combination a path is returned.

        The corresponding Graph instance has node annotations: The
        property "kind" is either set to "head", "tail" or "path".
        Edges are also annotated with "kind" -> "path". This is mainly
        used for visualizations.

        Parameters
        ----------

        h : int
          Head node

        t : int
          Tail node

        depth : int
          If greater than 0, also return neighbouring nodes

        cutoff: int
          Maximum depth to search for paths

        """
        assert h is not None
        assert t is not None

        gen = networkx.all_simple_paths(
            self.nx, source=h, target=t, cutoff=cutoff)

        visited = set()
        for nodes in map(tuple, gen):
            if nodes in visited:
                continue

            # ---------------------------------------- sub-graph

            g = Graph(name=f'{self.name}.p{h}-{t}_{cutoff}.{depth}')
            for node in nodes:
                _, subg = self.sample_subgraph(start=node, depth=depth)
                g.join(subg)

                # depth == 0 is a special case where the connecting
                # edges have to be added manually
                for edge in self.nx.edges([node], data=True):
                    h, t, _ = edge
                    if t in nodes and h != t:
                        self._safe_add(g, self, edge)

            # ---------------------------------------- paths

            for edges in self.connecting_edges(nodes):
                gclone = g.clone()._annotate_path(nodes, edges)
                yield Path(depth=depth, g=gclone, ents=nodes, rels=edges)

            visited.add(nodes)

    # ---

    def walk_agg_create(self, center: int):
        agg = WalkAggregator(walk=[center], pattern=[])
        agg.visited[center] += 1
        return agg

    def walk_agg_fork(self, agg: WalkAggregator, cutoff):
        node = agg.walk[-1]
        adj = list(self.find(heads={node}))

        exits = (
            # maximum depth reached
            len(agg.pattern) == cutoff,
            # cycle detected by re-occurrence
            agg.visited[node] > 1,
            # no outgoing edges
            not len(adj),
        )

        # any exit condition applies
        # calculate all possible sub-paths and
        # save them to the Stroll
        if any(exits):
            return

        assert len(agg.pattern) < cutoff

        aggs = []
        for _, t, r in adj:

            visited = agg.visited.copy()
            visited[t] += 1

            aggs.append(WalkAggregator(
                walk=agg.walk + [t],
                pattern=agg.pattern + [r],
                visited=visited,
            ))

        return aggs

    def stroll(self,
               center: int = None,
               cutoff: int = None) -> Stroll:
        """

        Samples all walks, patterns and grounds around a given graph center.


        Parameters
        ----------

        center : int
          start node to sample paths from

        cutoff : int
          maximum pattern length


        Returns
        -------

        All possible walks from the <center> of the subgraph of depth
        <cutoff>. The data structure is a mapping of patterns to
        sets of walks connecting <center> with <target> (where <target> is
        any node in the neighbourhood of <center>).

        Discussion
        ----------

        How cycles are handled is debatable...

        """
        assert center is not None
        assert cutoff is not None

        # log.info(
        #     f'strolling over {self.name}' +
        #     f'({self.source.ents[center]}) cutoff={cutoff}')

        stroll = Stroll()
        if cutoff == 0:
            return stroll

        # --

        q = queue.Queue()

        agg = self.walk_agg_create(center)
        q.put(agg)

        while not q.empty():
            agg = q.get()
            ret = self.walk_agg_fork(agg, cutoff)

            if ret is None:
                stroll.add(agg)
                continue

            for agg in ret:
                q.put(agg)

        return stroll

    #
    # --- | GRAPH-LEVEL OPERATIONS (in-place)
    #

    def join(self, other: 'Graph') -> 'Graph':
        """

        Add other graphs to this graph.
        See also the docs for networkx's compose().

        Parameters
        ----------

        other : Graph
          Other graph whose data is joined

        """
        self.source.join(other.source)
        self._nx = networkx.compose(self.nx, other.nx)
        return self

    def clone(self) -> 'Graph':
        g = Graph(name=self.name)

        g._source = self._source
        g._nx = self._nx.copy()

        return g

    #
    # --- | EXTERNAL SOURCES
    #

    def add(self, source: GraphImport) -> 'Graph':
        """

        Add data to the current graph by using a GraphImport instance

        Parameters
        ----------

        source : GraphImport
          Data to feed into the graph

        """
        for i, (h, t, r) in enumerate(source.triples):
            self.nx.add_node(h, label=source.ents[h])
            self.nx.add_node(t, label=source.ents[t])
            self.nx.add_edge(h, t, r, label=source.rels[r], rid=r)
            self.edges[r].add((h, t))

        self.source.join(source)
        self._rnx = self.nx.reverse()
        return self

    def save(self, f_name: str) -> 'Graph':
        """

        Persist graph to file.

        Parameters
        ----------

        f_name : str
          File to save the graph to

        """
        path = pathlib.Path(f_name)

        # TODO: catch error case
        # _relative = path.relative_to(ryn.ENV.ROOT_DIR)
        # log.info(f'saving graph {self.name} to {_relative}')

        with path.open(mode='wb') as fd:
            pickle.dump(self, fd)

        return self

    @staticmethod
    def load(f_name: str) -> 'Graph':
        """

        Load graph from file

        Parameters
        ----------

        f_name : str
          File to load graph from

        """
        path = pathlib.Path(f_name)
        log.info(f'loading graph from {path}')

        with path.open(mode='rb') as fd:
            return pickle.load(fd)

    #
    # ---| SUGAR
    #

    def tabulate_triples(self, triples):
        from tabulate import tabulate

        src = self.source

        rows = [(
            h, src.ents[h],
            t, src.ents[t],
            r, src.rels[r])
                for h, t, r in triples]

        return tabulate(rows, headers=('', 'head', '', 'tail', '', 'relation'))

    def str_triple(self, triple):
        h, t, r = triple
        return (
            f'{self.source.ents[h]} | '
            f'{self.source.ents[t]} | '
            f'{self.source.rels[r]}')


# ---


def _cli(args):
    from ryn.graphs import loader
    import IPython

    print()
    graphs = loader.load_graphs_from_args(args)
    for name in graphs:
        print(f'loaded graph: {name}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN GRAPH CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    graphs: Dict[str, ryn.graph.Graph]',
        '',
    ))

    IPython.embed(banner1=banner)
