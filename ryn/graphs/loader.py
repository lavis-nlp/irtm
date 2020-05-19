# --- | OPEN KE IMPORTER
#       https://github.com/thunlp/OpenKE


import ryn
from ryn import RynError
from ryn.graphs import graph
from ryn.common import config
from ryn.common import logging

import pathlib
import argparse
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Generator


log = logging.get('graphs.loader')


def _oke_fn_triples(line: str):
    h, t, r = map(int, line.split())
    return h, t, r


def _oke_fn_idmap(line: str):
    name, idx = line.rsplit(maxsplit=1)
    return int(idx), name.strip()


def _oke_parse(path: str = None, fn=None) -> Generator[Any, None, None]:
    if path is None:
        return None

    with open(path, mode='r') as fd:
        fd.readline()
        for i, line in enumerate(fd):
            yield line if fn is None else fn(line)


def load_oke(
        f_triples: str,
        f_rel2id: str = None,
        f_ent2id: str = None) -> graph.GraphImport:
    """

    Load OpenKE-like benchmark files. Structure is as follows:

    f_triples: the graph as (eid-1, eid-2, rid) triples
    f_rel2id: relation names as (name, rid) tuples
    f_ent2id: entity labels as (label, eid) tuples

    The first line of each file is ignored (contains the number of
    data points in the original data set)

    """
    log.info(f'loading OKE-like graph from {f_triples}')

    triples = tuple(_oke_parse(f_triples, _oke_fn_triples))
    rels = dict(_oke_parse(f_rel2id, _oke_fn_idmap))
    ents = dict(_oke_parse(f_ent2id, _oke_fn_idmap))

    gi = graph.GraphImport(triples=triples, rels=rels, ents=ents)

    log.info(f'finished parsing {f_triples}')

    return gi


# --- | VILLMOW IMPORTER
#       https://github.com/villmow/datasets_knowledge_embedding
#       https://gitlab.cs.hs-rm.de/jvill_transfer_group/thesis/thesis


def load_vll(f_triples: str) -> graph.GraphImport:
    """

    Load Villmow's benchmark files. Structure is as follows:

    f_triples: the graph encoded as string triples (e1, r, e2)

    """
    log.info(f'loading villmow-like graph from {f_triples}')

    triples = []
    refs = {
        'ents': {'counter': 0, 'dic': {}},
        'rels': {'counter': 0, 'dic': {}},
    }

    def _get(kind: str, key: str):
        dic = refs[kind]['dic']

        if key not in dic:
            dic[key] = refs[kind]['counter']
            refs[kind]['counter'] += 1

        return dic[key]

    with open(f_triples, mode='r') as fd:
        for line in fd:

            gen = zip(('ents', 'rels', 'ents'), line.strip().split())
            h, r, t = map(lambda a: _get(*a), gen)
            triples.append((h, t, r))  # mind the switch

    gi = graph.GraphImport(
        triples=triples,
        rels={idx: name for name, idx in refs['rels']['dic'].items()},
        ents={idx: name for name, idx in refs['ents']['dic'].items()},
    )

    log.info(f'finished parsing {f_triples}')

    return gi


LOADER = {
    'vll': load_vll,
    'oke': load_oke,
}


def load_graph(
        name: str = None,
        path: str = None,
        cache: str = None,
        reader: str = None,
        triples: str = None,
        entity_labels: str = None,
        relation_labels: str = None) -> Dict[str, graph.Graph]:

    assert name is not None
    assert path is not None
    assert cache is not None
    assert reader is not None
    assert triples is not None

    # ---

    log.info(f'loading graph "{name}"')

    cache_file = pathlib.Path(cache) / f'{name}.pkl'
    if cache_file.exists():
        log.info(f'loading cached "{cache_file}"')
        return graph.Graph.load(cache_file)

    # ---

    log.info('no cached file found, parsing')
    if entity_labels is None or relation_labels is None:
        files = (triples, )
    else:
        files = (triples, relation_labels, entity_labels)

    p_path = pathlib.Path(path)
    files = map(lambda p: p_path / pathlib.Path(p), files)

    source = LOADER[reader](*files)
    g = graph.Graph(name=name, source=source)

    # ---

    log.info(f'writing "{cache_file}"')
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    g.save(cache_file)

    return g


def load_graphs_from_conf(
        conf: str = None,
        spec: str = None,
        graphs: List[str] = None,
        single: bool = False) -> (
            Dict[str, graph.Graph]):

    if conf is None:
        raise RynError('provide a configuration file')

    if spec is None:
        raise RynError('provide a configuration specification')

    if graphs is None:
        raise RynError('provide a graph selection')

    # ---

    log.info(f'selecting config: {conf}')
    log.info(f'loading {graphs or "all"} from conf')

    gen = config.Config.create(conf, spec)
    confs = {c.name: c for c in gen}
    selection = set(graphs) & set(confs.keys())

    if not len(selection):
        msg = 'no graphs were selected:'
        options = ', '.join(confs.keys())

        exc = RynError(f'{msg}\n  Options: {options}')
        log.error(str(exc))
        raise exc

    if single and len(selection) != 1:
        exc = RynError('please select a single graph')
        log.error(str(exc))
        raise exc

    graph_dict = {}
    for name in selection:

        conf = confs[name].obj
        graph_dict[name] = load_graph(
            name=name,
            path=conf['path'],
            reader=conf['reader'],
            triples=conf['triples'],
            entity_labels=conf.get('entity labels', None),
            relation_labels=conf.get('relation labels', None),
            cache=ryn.ENV.CACHE_DIR / 'graphs.loader',
        )

    return graph_dict if not single else graph_dict[list(selection)[0]]


def load_graphs_from_args(args: argparse.Namespace, single: bool = False):
    return load_graphs_from_conf(
        conf=args.config,
        spec=args.spec,
        graphs=args.graphs,
        single=single)


def load_graphs_from_uri(*uris: str):
    """
    uri format: {provider}.{dataset}
    (e.g. vll.fb15k-237)

    all uris must have the same provider (for now)

    """
    providers, _ = zip(*(s.split('.') for s in uris))
    assert all(providers[0] == p for p in providers)
    provider = providers[0]

    spec = str(ryn.ENV.CONF_DIR / 'graphs.spec.conf')
    conf = str(ryn.ENV.CONF_DIR / f'graphs.{provider}.conf')

    graph_dict = load_graphs_from_conf(conf=conf, spec=spec, graphs=uris)

    # preserve order
    return [graph_dict[ds] for ds in uris]


# FIXME: find a way to make this non-global; take care of file_cache
_lazy_graph_loader_cache = {}


@dataclass(frozen=True)
class LazyGraphLoader:
    """

    Load graphs lazily and save them internally.

    This class is usable with the file_cache
    (which requires this class to be hashable) and avoids
    loading the whole graph structure.

    """

    config: str
    spec: str
    graphs: Tuple[str] = None

    def __post_init__(self):
        if self.graphs is not None:
            assert type(self.graphs) is tuple, type(self.graphs)

    def get(self, selection: Tuple[str] = None) -> Dict[str, graph.Graph]:
        cache = _lazy_graph_loader_cache
        names = selection or self.graphs
        graphs = {}

        for name in names:
            if name not in cache:
                cache[name] = load_graphs_from_conf(
                    self.config, self.spec, (name, ))[name]

            graphs[name] = cache[name]

        return graphs

    @staticmethod
    def from_args(
            args: argparse.Namespace,
            selection: Tuple[str] = None) -> 'LazyGraphLoader':

        return LazyGraphLoader(
            config=args.config,
            spec=args.spec,
            graphs=selection)


def add_graph_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        '-c', '--config', type=str,
        help='config file (conf/*.conf)',
    )

    parser.add_argument(
        '-s', '--spec', type=str,
        help='config specification file (conf/*.spec.conf)',
    )

    parser.add_argument(
        '-g', '--graphs', type=str, nargs='+',
        help='selection of graphs (names defined in --config)'
    )
