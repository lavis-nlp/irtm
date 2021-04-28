# -*- coding: utf-8 -*-

from irtm.common import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

import dataclasses


log = logging.get('common.plotter')

# used for display in emacs notebooks with dark background
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"


CLR = ['#333333', '#FF3E72', '#2BB1FF', '#15587F', '#FFDDAB', '#B59D79']


@dataclasses.dataclass
class Plotter:

    title: str

    xlabel: str = None
    ylabel: str = None

    display: bool = True
    fname:    str = None

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    def __post_init__(self):
        self._fig = plt.figure()
        self._ax = self.fig.add_subplot(111)

        self.ax.set_title(self.title)
        self.xlabel and self.ax.set_xlabel(self.xlabel)
        self.ylabel and self.ax.set_ylabel(self.ylabel)

    def plot(self, patches=None):
        if patches is not None:
            self.ax.legend(handles=patches)

        if self.display:
            plt.show(self.fig)

        if self.fname is not None:
            log.info(f'saving figure to "{self.fname}"')
            for suff in ('.png', '.svg', '.eps'):
                self.fig.savefig(self.fname + suff)

        plt.close(self.fig)
