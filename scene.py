from manim import *


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()                   # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))     # show the circle on screen


class ThreeDFunctionPlot(ThreeDScene):
    def construct(self):
        resolution_fa = 22
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_plane(u, v):
            x = u
            y = v
            z = 0
            return np.array([x, y, z])

        plane = ParametricSurface(
            param_plane,
            resolution=(resolution_fa, resolution_fa),
            v_min=-2,
            v_max=+2,
            u_min=-2,
            u_max=+2,
        )
        plane.scale_about_point(2, ORIGIN)

        def param_gauss(u, v):
            x = u
            y = v
            d = np.sqrt(x * x + y * y)
            sigma, mu = 0.4, 0.0
            z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = ParametricSurface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_min=-2,
            v_max=+2,
            u_min=-2,
            u_max=+2,
        )

        gauss_plane.scale_about_point(2, ORIGIN)
        gauss_plane.set_style(fill_opacity=1)
        gauss_plane.set_style(stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(GREEN, BLUE, opacity=0.1)

        axes = ThreeDAxes()

        self.add(axes)
        self.play(Write(plane))
        self.play(Transform(plane, gauss_plane))
        self.wait()

class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeInFrom(basel, UP),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOutAndShift(obj, direction=DOWN) for obj in basel]),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid")
        grid_title.scale(1.5)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeInFrom(grid_title, direction=DOWN),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid"
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                          + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ]
                )
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()