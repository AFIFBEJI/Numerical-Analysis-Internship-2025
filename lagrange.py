from manim import *
import numpy as np
import sympy as sp
from scipy.interpolate import lagrange

class DataTable:
    """Class to handle data table creation and animation."""
    
    def __init__(self, scene, jours, prix):
        self.scene = scene
        self.jours = jours
        self.prix = prix
        
    def create_table(self):
        # Create table data structure
        table_data = [
            ["Jours", "Prix"],  # Headers
            *[[str(j), f"{p:.1f}"] for j, p in zip(self.jours, self.prix)]
        ]
        
        # Create the table object
        table = Table(table_data, include_outer_lines=True).scale(0.5)
        return table
    
    def animate_table(self, table):
        # Show table with animation
        self.scene.play(Create(table))
        self.scene.wait(2)
        
        # Move table to the right
        self.scene.play(table.animate.shift(RIGHT * 3.5))
        self.scene.wait(2)
        
        return table


class GraphPlotter:
    """Class to handle graph creation and plotting."""
    
    def __init__(self, scene, jours, prix):
        self.scene = scene
        self.jours = jours
        self.prix = prix
        
    def create_axes(self):
        # Create axes for the graph
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[90, 115, 5],
            axis_config={"color": WHITE},
            x_length=5,
            y_length=4,
        ).to_edge(LEFT)
        
        # Create axis labels with properly escaped dollar sign
        labels = axes.get_axis_labels(x_label="Jours", y_label=r"Prix (\$)")
        
        return axes, labels
    
    def animate_axes(self, axes, labels):
        # Display axes and labels
        self.scene.play(Create(axes), Write(labels))
        self.scene.wait(1)
        
        # Add tick values for X axis
        x_ticks = VGroup(*[Text(str(i), font_size=24).next_to(axes.c2p(i, 0), DOWN * 0.3) for i in range(2, 6)])
        
        # Add tick values for Y axis
        y_ticks = VGroup(*[Text(str(i), font_size=24).next_to(axes.c2p(0, i), LEFT * 0.3) for i in range(90, 116, 5)])
        
        # Display tick values
        self.scene.play(Write(x_ticks), Write(y_ticks))
        self.scene.wait(2)
        
        return x_ticks, y_ticks
    
    def plot_data_points(self, axes):
        # Plot data points without the Lagrange curve
        point_dots = VGroup(*[Dot(axes.c2p(j, p), color=YELLOW) for j, p in zip(self.jours, self.prix)])
        point_labels = VGroup(*[Text(f"{p:.1f}$", font_size=24).next_to(Dot(axes.c2p(j, p)), UP) 
                              for j, p in zip(self.jours, self.prix)])
        
        # Animate points and labels appearance
        self.scene.play(Create(point_dots), Write(point_labels))
        self.scene.wait(2)
        
        return point_dots, point_labels
    
    def lagrange_interpolation(self, x_val):
        """Calculate the Lagrange interpolation value at x_val."""
        result = 0
        for i in range(len(self.jours)):
            term = self.prix[i]
            for j in range(len(self.jours)):
                if i != j:
                    term *= (x_val - self.jours[j]) / (self.jours[i] - self.jours[j])
            result += term
        return result
    
    def plot_lagrange_curve(self, axes):
        # Plot Lagrange interpolation curve
        lagrange_graph = axes.plot(
            lambda x: self.lagrange_interpolation(x),
            x_range=[1, 4],
            color=BLUE
        )
        
        # Show the interpolation curve
        self.scene.play(Create(lagrange_graph))
        self.scene.wait(2)
        
        return lagrange_graph
    
    def animate_price_estimation(self, axes, x_value):
        # Create cursor (red dot)
        cursor = Dot(axes.c2p(1, self.prix[0]), color=RED)
        self.scene.play(FadeIn(cursor))
        
        # Target point for the cursor
        target_y = self.lagrange_interpolation(x_value)
        
        # Animate cursor moving from x=1 to x=x_value
        self.scene.play(cursor.animate.move_to(axes.c2p(x_value, target_y)), run_time=3)
        
        # Add vertical line at x_value
        vertical_line = DashedLine(
            start=axes.c2p(x_value, 90),
            end=axes.c2p(x_value, target_y),
            color=WHITE
        )
        self.scene.play(Create(vertical_line))
        self.scene.wait(1)
        
        # Show estimated price label
        price_label = Text(f"{target_y:.1f} $", font_size=24, color=YELLOW).next_to(cursor, UP * 0.5 + RIGHT)
        self.scene.play(Write(price_label))
        self.scene.wait(2)
        
        return cursor, vertical_line, price_label


class TextManager:
    """Class to handle text creation and animation."""
    
    def __init__(self, scene):
        self.scene = scene
    
    def show_question(self):
        lines = [
            "How can we represent these price",
            "variations as a continuous curve that",
            "passes exactly through these points",
            "and helps estimate the market",
            "trend?"
        ]
        
        text_objects = VGroup()
        prev_line = None
        
        for i, line_text in enumerate(lines):
            line = Text(line_text, font_size=25, color=RED)
            
            if i == 0:
                line.to_edge(RIGHT).shift(UP)
            else:
                line.next_to(text_objects[-1], DOWN, buff=0.3)
                
            text_objects.add(line)
            self.scene.play(Write(line))
        
        self.scene.wait(2)
        self.scene.play(FadeOut(text_objects))
        
        return text_objects
    
    def show_answer(self):
        lines = [
            "We can use Lagrange interpolation,",
            "a mathematical method that constructs a ",
            "polynomial passing exactly through these ",
            "points, helping to visualize trends",
            "and estimate market variations."
        ]
        
        text_objects = VGroup()
        
        for i, line_text in enumerate(lines):
            line = Text(line_text, font_size=25, color=GREEN_A)
            
            if i == 0:
                line.to_edge(RIGHT).shift(UP)
            else:
                line.next_to(text_objects[-1], DOWN, buff=0.3)
                
            text_objects.add(line)
            self.scene.play(Write(line))
        
        self.scene.wait(1)
        
        return text_objects
    
    def show_title(self):
        title = Text("Polynomial Interpolation", color=BLUE).to_edge(UP).scale(1)
        subtitle = Text("Lagrange Method", color=GREEN).next_to(title, DOWN, buff=0.5).scale(0.8)
        
        # Animation: Show title and subtitle
        self.scene.play(Write(title))
        self.scene.wait(1)
        self.scene.play(Write(subtitle))
        self.scene.wait(1)
        
        # Move title up and replace with subtitle
        self.scene.play(
            title.animate.to_edge(UP).shift(UP * 3),
            subtitle.animate.move_to(title.get_center())
        )
        self.scene.wait(1)
        
        return title, subtitle
    
    def show_thinking(self, subtitle):
        thinking = Text("Alex wonders what the stock price might be on day 3.5.", font_size=24)
        thinking.next_to(subtitle, DOWN, buff=0.5)
        self.scene.play(Write(thinking))
        self.scene.wait(2)
        
        return thinking


class LagrangeFormulas:
    """Class to handle Lagrange formula creation and animation."""
    
    def __init__(self, scene, jours, prix):
        self.scene = scene
        self.jours = jours
        self.prix = prix
        self.x = sp.Symbol('x')
    
    def show_theorems(self, subtitle):
        # Define Theorem 1
        theorem1 = MathTex(r"P_n(x)", r"= \sum_{i=0}^{n} y_i L_i(x)").scale(0.9)
        theorem1.set_color_by_tex(r"P_n(x)", YELLOW)
        theorem1.next_to(subtitle, RIGHT * 0.05 + DOWN, buff=0.5)
        
        # Define domain condition
        domain = MathTex(r"\text{where } x \in \mathbb{R}").scale(0.8)
        domain.next_to(theorem1, DOWN, buff=0.5)
        
        # Define Theorem 2
        theorem2 = MathTex(
            r"L_i(x)", r"= \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}"
        ).scale(0.9)
        theorem2.set_color_by_tex(r"L_i(x)", RED)
        theorem2.next_to(domain, DOWN, buff=0.5)
        
        # Animate Theorems
        self.scene.play(Write(theorem1))
        self.scene.wait(1)
        self.scene.play(Write(domain))
        self.scene.wait(1)
        self.scene.play(Write(theorem2))
        self.scene.wait(2)
        
        self.scene.play(
            FadeOut(domain),
            theorem2.animate.move_to(domain.get_center())
        )
        
        return theorem1, theorem2
    
    def lagrange_basis(self, i):
        """Calculate the Lagrange basis polynomial for index i."""
        terms = [(self.x - self.jours[j]) / (self.jours[i] - self.jours[j]) 
                for j in range(len(self.jours)) if j != i]
        return sp.simplify(sp.prod(terms))
    
    def show_basis_polynomials(self, theorem2):
        # Calculate and display L_i(x)
        L = [self.lagrange_basis(i) for i in range(len(self.jours))]
        
        # Show each L_i(x) in Manim
        equations = VGroup()
        for i, Li in enumerate(L):
            eq = MathTex(f"L_{i}(x) =", sp.latex(Li), font_size=28)
            if i == 0:
                eq.next_to(theorem2, DOWN, buff=0.3)
            else:
                eq.next_to(equations[-1], DOWN, buff=0.3)
            equations.add(eq)
            self.scene.play(Write(eq))
            self.scene.wait(1)
        
        # Show simplified versions of each L_i(x)
        simplified_equations = VGroup()
        for i, Li in enumerate(L):
            Li_simplified = sp.expand(Li)
            eq_simplified = MathTex(f"L_{i}(x) =", sp.latex(Li_simplified), font_size=28)
            eq_simplified.set_color(BLUE)
            eq_simplified.move_to(equations[i].get_center())
            simplified_equations.add(eq_simplified)
            self.scene.play(Transform(equations[i], eq_simplified))
            self.scene.wait(1)
        
        # Hide theorem2
        self.scene.play(FadeOut(theorem2))
        self.scene.wait(1)
        
        # Move L_i(x) equations up
        self.scene.play(equations.animate.shift(UP * 1.5))
        self.scene.wait(1)
        
        return L, equations
    
    def show_polynomial(self, equations, theorem1):
        # Show P_n(x) in explicit form
        poly = MathTex(
            r"P_n(x) = y_0 L_0(x) + y_1 L_1(x) + y_2 L_2(x) + y_3 L_3(x)",
            font_size=28
        ).set_color(YELLOW)
        poly.next_to(equations[-1], DOWN, buff=0.5)
        
        self.scene.play(Write(poly))
        self.scene.wait(2)
        
        # Show P_n(x) with numerical values
        poly_values = MathTex(
            f"P_n(x) = {self.prix[0]} L_0(x) + {self.prix[1]} L_1(x) + {self.prix[2]} L_2(x) + {self.prix[3]} L_3(x)",
            font_size=28
        ).next_to(poly, DOWN, buff=0.5).align_to(poly, LEFT)
        
        self.scene.play(Write(poly_values))
        self.scene.wait(2)
        
        return poly, poly_values
    
    def show_expanded_polynomial(self, L, poly, poly_values, equations, theorem1):
        # Calculate final P_n(x)
        P_n = sum(self.prix[i] * L[i] for i in range(len(self.jours)))
        P_n_simplified = sp.expand(P_n)
        
        # Hide L_i(x) equations
        self.scene.play(FadeOut(equations))
        self.scene.play(FadeOut(poly))
        self.scene.play(FadeOut(theorem1))
        
        # Show simplified P_n(x)
        pn_equation = MathTex(r"P_n(x) =", sp.latex(P_n_simplified), font_size=30)
        pn_equation.set_color(GREEN)
        pn_equation.move_to(RIGHT * 3)
        
        # Transform to simplified equation
        self.scene.play(Transform(poly_values, pn_equation))
        
        return P_n_simplified, poly_values


class Data(Scene):
    def construct(self):
        # Initialize data
        jours = np.array([1, 2, 3, 4])
        prix = np.array([98, 105, 95, 110])
        
        # Initialize helper classes
        data_table = DataTable(self, jours, prix)
        graph_plotter = GraphPlotter(self, jours, prix)
        text_manager = TextManager(self)
        lagrange_formulas = LagrangeFormulas(self, jours, prix)
        
        # 1. Create and animate table
        table = data_table.create_table()
        data_table.animate_table(table)
        
        # 2. Create axes and plot data points
        axes, labels = graph_plotter.create_axes()
        x_ticks, y_ticks = graph_plotter.animate_axes(axes, labels)
        point_dots, point_labels = graph_plotter.plot_data_points(axes)
        
        # 3. Hide table and show question
        self.play(FadeOut(table))
        text_manager.show_question()
        
        # 4. Show answer and title
        answer_lines = text_manager.show_answer()
        title, subtitle = text_manager.show_title()
        
        # 5. Hide answer and show theorems
        self.play(FadeOut(answer_lines))
        theorem1, theorem2 = lagrange_formulas.show_theorems(subtitle)
        
        # 6. Show Lagrange basis polynomials
        L, equations = lagrange_formulas.show_basis_polynomials(theorem2)
        
        # 7. Show polynomial formulations
        poly, poly_values = lagrange_formulas.show_polynomial(equations, theorem1)
        
        # 8. Calculate and show expanded polynomial
        P_n, pn_equation = lagrange_formulas.show_expanded_polynomial(L, poly, poly_values, equations, theorem1)
        
        # 9. Plot Lagrange curve
        lagrange_graph = graph_plotter.plot_lagrange_curve(axes)
        
        # 10. Hide equation and prepare for estimation
        self.play(FadeOut(pn_equation))
        
        # 11. Show Alex's question and estimate price
        thinking = text_manager.show_thinking(subtitle)
        cursor, vertical_line, price_label = graph_plotter.animate_price_estimation(axes, 3.5)
        
        # 12. Clean up scene
        self.play(FadeOut(cursor, vertical_line, price_label, thinking, lagrange_graph))
        self.wait(2)