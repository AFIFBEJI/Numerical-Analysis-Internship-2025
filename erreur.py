from manim import *
import numpy as np

class FormulaExplainer:
    """Class to handle the creation and animation of mathematical formulas."""
    
    def __init__(self, scene):
        self.scene = scene
    
    def create_title(self):
        title = Text("Interpolation Error Estimation", font_size=36)
        title.to_edge(UP)
        return title
    
    def create_formula(self):
        formula = MathTex(
            r"E_n(x) \leq "
            r"\frac{\max\limits_{t \in [x_0, x_n]} \left| f^{(n+1)}(t) \right|}"
            r"{(n+1)!} \prod\limits_{i=0}^{n} \left| x - x_i \right|"
        )
        formula.scale(0.9)
        return formula
    
    def get_term_explanations(self):
        return [
            ("\max_{t \in [x_0, x_n]} |f^{(n+1)}(t)|", "Upper bound of the derivative of f"),
            ("(n+1)!", "Factorial of (n+1)"),
            ("\prod_{i=0}^{n} |x - x_i|", "Product of distances to interpolation points")
        ]
    
    def show_explanations(self, formula):
        for term, explanation in self.get_term_explanations():
            term_mob = MathTex(term)
            term_mob.set_color(YELLOW)
            term_mob.next_to(formula, DOWN, buff=0.5)

            explanation_mob = Text(explanation, font_size=24)
            explanation_mob.next_to(term_mob, DOWN, buff=0.3)

            self.scene.play(Indicate(formula))
            self.scene.play(TransformMatchingShapes(formula.copy(), term_mob))
            self.scene.play(Write(explanation_mob))
            self.scene.wait(2)

            self.scene.play(FadeOut(term_mob, explanation_mob))


class FunctionPlotter:
    """Class to handle plotting of functions and visualizing errors."""
    
    def __init__(self, scene):
        self.scene = scene
        self.error_threshold = 0.5
        
    def create_axes(self):
        axes = Axes(
            x_range=[-2, 2, 0.5], 
            y_range=[-0.5, 8, 1],
            axis_config={"color": WHITE}
        ).scale(0.7).move_to(DOWN * 0.5)
        
        labels = axes.get_axis_labels(x_label="x", y_label="y").scale(0.7)
        return axes, labels
    
    def create_functions(self, axes):
        # Original function and its interpolation
        exp_func = axes.plot(lambda x: np.exp(x), color=BLUE)
        interp_func = axes.plot(lambda x: 1 + x + (x**2)/2, color=GREEN)
        
        # Create function labels
        exp_label = MathTex(r"f(x) = e^x").set_color(BLUE).scale(0.7)
        interp_label = MathTex(r"P(x) = 1 + x + \frac{x^2}{2}").set_color(GREEN).scale(0.7)
        
        # Position labels
        exp_label.next_to(axes.c2p(2, np.exp(2)), RIGHT, buff=0.5)
        interp_label.next_to(axes.c2p(1.5, 1 + 1.5 + (1.5**2)/2), DOWN * 0.4 + RIGHT, buff=0.3)
        
        return exp_func, interp_func, exp_label, interp_label
    
    def create_error_visualization(self, axes):
        error_points = VGroup()
        error_lines = VGroup()
        
        # Generate points where error exceeds threshold
        for x in np.linspace(-2, 2, 100):
            f_x = np.exp(x)
            p_x = 1 + x + (x**2)/2
            error = abs(f_x - p_x)
            
            if error > self.error_threshold:
                point_f = axes.c2p(x, f_x)
                point_p = axes.c2p(x, p_x)
                
                dot = Dot(point_f, color=RED)
                line = Line(start=point_f, end=point_p, color=YELLOW)
                
                error_points.add(dot)
                error_lines.add(line)
                
        return error_points, error_lines


class InterpolationError(Scene):
    def construct(self):
        # Initialize helper classes
        formula_explainer = FormulaExplainer(self)
        function_plotter = FunctionPlotter(self)
        
        # Create and display title
        title = formula_explainer.create_title()
        self.play(Write(title))
        self.wait(1)
        
        # Create and display formula
        formula = formula_explainer.create_formula()
        formula.next_to(title, DOWN, buff=0.5)
        self.play(Write(formula))
        self.wait(2)
        
        # Show formula explanations
        formula_explainer.show_explanations(formula)
        self.play(FadeOut(formula))
        self.wait(2)
        
        # Plot functions and show error
        axes, labels = function_plotter.create_axes()
        exp_func, interp_func, exp_label, interp_label = function_plotter.create_functions(axes)
        error_points, error_lines = function_plotter.create_error_visualization(axes)
        
        # Animate everything
        self.play(Create(axes), Write(labels))
        self.play(Create(exp_func), Write(exp_label))
        self.play(Create(interp_func), Write(interp_label))
        self.play(Create(error_points))
        self.play(Create(error_lines))
        
        self.wait(6)
