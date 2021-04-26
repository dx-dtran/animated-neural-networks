from manim import *
import csv
import numpy as np


def return_coords_from_csv(file_name):
    coords = []
    with open(f"{file_name}.csv", "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            x, y, label = row
            coord = [float(x), float(y)]
            coords.append(coord)
    return coords


class CSV(GraphScene):
    def __init__(self):
        GraphScene.__init__(
            self,
            x_min=-6,
            x_max=6,
            y_min=-6,
            y_max=6,
            graph_origin=ORIGIN
        )

    def construct(self):
        self.setup_axes()
        coords, labels = return_coords_from_csv("sample_data/xor_data")
        colors = [ORANGE if label == "-1" else BLUE for label in labels]
        dots = VGroup(
            *[
                Dot(color=col).move_to(self.coords_to_point(coord[0], coord[1]))
                for coord, col in zip(coords, colors)
            ]
        )
        self.add(dots)
        self.play(Create(dots), run_time=3)


class ThreeDScatterPlot(ThreeDScene):
    def construct(self):
        coords, labels = return_coords_from_csv("sample_data/xor_data")
        colors = [ORANGE if label == "-1" else BLUE for label in labels]
        axes = ThreeDAxes()
        dots = VGroup(
            *[
                Dot3D(color=col).move_to(
                    axes.coords_to_point(coord[0], coord[1], 0))
                for coord, col in zip(coords, colors)
            ]
        )
        self.add(axes, dots)
        self.play(Create(dots), run_time=3)


class DotsMoving(Scene):
    def construct(self):
        dots = [Dot() for i in range(5)]
        directions = [np.random.randn(3) for dot in dots]
        self.add(*dots) # It isn't absolutely necessary
        animations = [ApplyMethod(dot.shift,direction) for dot,direction in zip(dots,directions)]
        self.play(*animations) # * -> unpacks the list animations
