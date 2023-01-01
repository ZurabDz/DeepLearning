"""
Basically poor man's tensorboard...
"""

import collections
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display


def use_svg_display():
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class ProgressBoard:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.ls = ls
        self.colors = colors
        self.fig = fig
        self.axes = axes
        self.figsize = figsize
        self.display = display

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple("Point", ["x", "y"])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(
                plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[
                    0
                ]
            )
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim:
            axes.set_xlim(self.xlim)
        if self.ylim:
            axes.set_ylim(self.ylim)
        if not self.xlabel:
            self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)
