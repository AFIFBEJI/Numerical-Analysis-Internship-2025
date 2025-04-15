from manim import *

class TitleManager:
    """Class to handle title creation and animation."""
    
    def __init__(self, scene):
        self.scene = scene
    
    def create_and_show_title(self):
        title = Text("Limitations of Lagrange Interpolation", 
                     font_size=40, color=BLUE).to_edge(UP)
        
        self.scene.play(Write(title))
        self.scene.wait(0.5)
        
        return title


class LimitationsManager:
    """Class to handle limitation points creation and animation."""
    
    def __init__(self, scene):
        self.scene = scene
        self.limitations = []
        self.descriptions = []
    
    def create_limitations(self, title):
        # Limitation 1
        point1 = Text("1. No Incremental Computation", 
                      font_size=30, color=YELLOW).next_to(title, DOWN, buff=0.5)
        desc1 = Tex(r"$P_{n+1}(x)$ cannot be derived from $P_n(x)$.", 
                    font_size=36).next_to(point1, DOWN, buff=0.3)
        
        # Limitation 2
        point2 = Text("2. Oscillations with More Points", 
                      font_size=30, color=ORANGE).next_to(desc1, DOWN, buff=0.5)
        desc2 = Text("More points → More errors (Runge's Phenomenon).", 
                     font_size=26).next_to(point2, DOWN, buff=0.3)
        
        # Limitation 3
        point3 = Text("3. Computational Complexity", 
                      font_size=30, color=RED).next_to(desc2, DOWN, buff=0.5)
        desc3 = Text("Large datasets make calculations inefficient.", 
                     font_size=26).next_to(point3, DOWN, buff=0.3)
        
        self.limitations = [point1, point2, point3]
        self.descriptions = [desc1, desc2, desc3]
        
        return self.limitations, self.descriptions
    
    def show_limitations(self):
        for point, desc in zip(self.limitations, self.descriptions):
            self.scene.play(Write(point))
            self.scene.play(Write(desc))
            self.scene.wait(0.5)
        
        self.scene.wait(2)
    
    def hide_limitations(self):
        for point, desc in zip(self.limitations, self.descriptions):
            self.scene.play(FadeOut(point))
            self.scene.play(FadeOut(desc))
            self.scene.wait(0.5)


class RungeVisualization:
    """Class to handle visualization of Runge's phenomenon."""
    
    def __init__(self, scene):
        self.scene = scene
    
    def create_and_show_runge(self):
        # Create title
        title = Text("Runge's Phenomenon", font_size=36, color=ORANGE).to_edge(UP)
        
        # Create axes
        axes = Axes(
            x_range=[-1, 1, 0.2],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"color": WHITE},
            x_length=8,
            y_length=5
        ).shift(DOWN * 0.5)
        
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Create Runge function
        runge_func = lambda x: 1 / (1 + 25 * x**2)
        runge_graph = axes.plot(runge_func, color=BLUE, x_range=[-1, 1, 0.01])
        
        runge_label = Text("Runge Function: f(x) = 1/(1+25x²)", 
                           font_size=24).next_to(axes, UP, buff=0.3)
        
        # Animate elements
        self.scene.play(Write(title))
        self.scene.play(Create(axes), Write(labels))
        self.scene.play(Create(runge_graph), Write(runge_label))
        
        # Create data points for interpolation
        n_points = 11
        x_points = np.linspace(-1, 1, n_points)
        y_points = [runge_func(x) for x in x_points]
        
        dots = VGroup(*[Dot(axes.c2p(x, y), color=YELLOW) 
                       for x, y in zip(x_points, y_points)])
        
        # Create interpolation polynomial
        coeffs = np.polyfit(x_points, y_points, n_points-1)
        poly_func = np.polynomial.polynomial.Polynomial(coeffs[::-1])
        
        interp_graph = axes.plot(
            lambda x: poly_func(x),
            color=RED,
            x_range=[-1, 1, 0.01]
        )
        
        # Animate interpolation
        self.scene.play(Create(dots))
        self.scene.wait(1)
        self.scene.play(Create(interp_graph))
        
        # Create legend
        legend = VGroup(
            Dot(color=BLUE).scale(1.5),
            Text("Original function", font_size=20, color=BLUE),
            Dot(color=RED).scale(1.5),
            Text("Interpolation polynomial", font_size=20, color=RED)
        ).arrange(RIGHT, buff=0.2).to_edge(DOWN, buff=0.3)
        
        self.scene.play(Create(legend))
        self.scene.wait(2)
        
        # Cleanup
        self.scene.play(
            FadeOut(title, axes, labels, runge_graph, runge_label, 
                    dots, interp_graph, legend)
        )
        
        return title, axes, runge_graph, interp_graph


class LagrangeLimitations(Scene):
    def construct(self):
        # Initialize helper classes
        title_manager = TitleManager(self)
        limitations_manager = LimitationsManager(self)
        runge_visualization = RungeVisualization(self)
        
        # 1. Create and show main title
        title = title_manager.create_and_show_title()
        
        # 2. Create and show limitations
        limitations, descriptions = limitations_manager.create_limitations(title)
        limitations_manager.show_limitations()
        
        # 3. Hide limitations
        limitations_manager.hide_limitations()
        
        # 4. Show Runge's Phenomenon visualization
        runge_visualization.create_and_show_runge()
        
        # 5. Clean up
        self.play(FadeOut(title))
        self.wait(2)

