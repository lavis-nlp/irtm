# -*- coding: utf-8 -*-

from ryn.common import config
from ryn.common import logging

from OpenKE.openke import data as oke_data
from OpenKE.openke import config as oke_config
from OpenKE.openke.module import model as oke_model

from dataclasses import dataclass

log = logging.get('embers.oke')


models = {
    'analogy': oke_model.Analogy,
    'transd': oke_model.TransD,
    'transe': oke_model.TransE,
    'transh': oke_model.TransH,
    'transr': oke_model.TransR,
    'rescal': oke_model.RESCAL,
    'simple': oke_model.SimplE,
    'complex': oke_model.ComplEx,
    'distmult': oke_model.DistMult,
}


@dataclass
class DataLoaderConfig:

    in_path: str
    nbatches: int

    threads: int
    sampling_mode: str

    bern_flag: int
    filter_flag: int

    neg_ent: int
    neg_rel: int

    @classmethod
    def fields(K):
        return tuple(K.__dataclass_fields__.keys())

    @classmethod
    def from_conf(K, conf: config.Config):
        params = {k: conf.obj['OpenKE'][k.replace('_', ' ')] for k in K.fields()}
        return K(**params)


def train(conf: config.Config):

    dataloader_conf = DataLoaderConfig.from_conf(conf)
    train_dataloader = oke_data.TrainDataLoader(**dataloader_conf.__dict__)

    __import__("IPython").embed(); __import__("sys").exit()

# distmult = DistMult(
# 	ent_tot = train_dataloader.get_ent_tot(),
# 	rel_tot = train_dataloader.get_rel_tot(),
# 	dim = 200
# )

# define the loss function
# model = NegativeSampling(
# 	model = distmult,
# 	loss = SoftplusLoss(),
# 	batch_size = train_dataloader.get_batch_size(),
# 	regul_rate = 1.0
# )

# train the model
# trainer = Trainer(model = model, data_loader = train_dataloader,
# train_times = 2000, alpha = 0.5, use_gpu = True, opt_method =
# "adagrad")
# trainer.run()
# distmult.save_checkpoint('./checkpoint/distmult.ckpt')

# ---


def run(exp: config.Config):
    msg = f'configuring experiment "{exp.name}"'
    print(f'\n{msg}\n')
    log.info(f'✝ {msg}')

    model = exp.obj["Ryn"]["model"]
    log.info(f'✝ training: {exp.name}: {model}')

    train(exp)
    log.info(f'✝ finished: {exp.name}: {model}')


def train_from_args(args):
    log.info('running OpenKE training')
    config.Config.execute(fconf=args.config, fspec=args.spec, callback=run)
