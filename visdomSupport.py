# encoding=utf-8
import visdom
from config import opt

vis = visdom.Visdom(opt.port)


def visPlot(x, y):
    vis.line(x, y)
