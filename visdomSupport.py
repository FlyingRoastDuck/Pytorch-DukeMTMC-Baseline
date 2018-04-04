# encoding=utf-8
import visdom

vis = visdom.Visdom


def visPlot(x, y):
    vis.line(x, y)
