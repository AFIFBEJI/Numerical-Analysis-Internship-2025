from manim import *
import numpy as np
import sympy as sp

class NewtonCalculator:
    """Class to handle Newton interpolation calculations."""
    
    def __init__(self, jours, prix):
        self.jours = jours
        self.prix = prix
        
    def newton_coefficients(self):
        """Calculate Newton coefficients using divided differences."""
        n = len(self.jours)
        coef = np.copy(self.prix).astype(float)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (self.jours[i] - self.jours[i - j])
        return coef
    
    def newton_interpolation(self, x):
        """Calculate interpolated value at point x."""
        coef = self.newton_coefficients()
        n = len(coef)
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x - self.jours[i]) + coef[i]
        return result

class GraphPlotter:
    """Class to handle graph creation and plotting."""
    
    def __init__(self, scene, jours, prix):
        self.scene = scene
        self.jours = jours
        self.prix = prix
        self.calculator = NewtonCalculator(jours, prix)
        
    def create_axes(self):
        """Create and return the axes for the graph."""
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[90, 110, 5],
            axis_config={"color": WHITE},
            x_length=5,
            y_length=4,
        ).to_edge(LEFT)
        
        labels = axes.get_axis_labels(x_label="Jours", y_label=r"Prix (\$)")
        return axes, labels
    
    def plot_data_points(self, axes):
        """Plot the data points on the graph."""
        point_dots = VGroup(*[Dot(axes.c2p(j, p), color=YELLOW) 
                            for j, p in zip(self.jours, self.prix)])
        point_labels = VGroup(*[Text(f"{p:.1f}$", font_size=24)
                              .next_to(Dot(axes.c2p(j, p)), UP) 
                              for j, p in zip(self.jours, self.prix)])
        
        self.scene.play(Create(point_dots), Write(point_labels))
        self.scene.wait(2)
        return point_dots, point_labels
    
    def plot_newton_curve(self, axes):
        """Plot the Newton interpolation curve."""
        newton_graph = axes.plot(
            lambda x: self.calculator.newton_interpolation(x),
            x_range=[1, 3],
            color=BLUE
        )
        
        self.scene.play(Create(newton_graph))
        self.scene.wait(2)
        return newton_graph

class TextManager:
    """Class to handle text creation and animation."""
    
    def __init__(self, scene):
        self.scene = scene
    
    def show_title(self):
        """Show the main title and subtitle."""
        title = Text("Polynomial Interpolation", color=BLUE).to_edge(UP).scale(1)
        subtitle = Text("Newton's Method", color=GREEN).next_to(title, DOWN, buff=0.5).scale(0.8)
        
        self.scene.play(Write(title))
        self.scene.wait(1)
        self.scene.play(Write(subtitle))
        self.scene.wait(1)
        
        self.scene.play(
            title.animate.to_edge(UP).shift(UP * 3),
            subtitle.animate.move_to(title.get_center())
        )
        self.scene.wait(1)
        
        return title, subtitle
    
    def show_conditions(self, subtitle):
        """Show the interpolation conditions."""
        condition_text = MathTex(
            r"x_i \neq x_j \quad \text{for } i \neq j", font_size=35
        ).next_to(subtitle, DOWN + RIGHT, buff=0.3).set_color(GREEN)
        
        equation1 = MathTex(
            r"P_n(x_i) = y_i, \quad \forall i \in \{0,1,\dots,n\}",
            font_size=35
        ).next_to(condition_text, DOWN, buff=0.3).set_color(GREEN)
        
        self.scene.play(Write(condition_text))
        self.scene.wait(1)
        self.scene.play(Write(equation1))
        self.scene.wait(2)
        
        return condition_text, equation1
    
    def show_polynomial_form(self, equation1):
        """Show the polynomial form and explanation."""
        polynomial_text = Text(
            "The polynomial can be written as:", font_size=24, color=WHITE
        ).next_to(equation1, DOWN, buff=0.3)
        
        polynomial_formula = MathTex(
            r"P_n(x) = \sum_{i=0}^{n} \beta_i \omega_i(x)", font_size=40
        ).next_to(polynomial_text, DOWN, buff=0.3).set_color(YELLOW)
        
        self.scene.play(Write(polynomial_text))
        self.scene.play(Write(polynomial_formula))
        self.scene.wait(2)
        
        return polynomial_text, polynomial_formula

def show_title_and_subtitle(scene):
    """Show the main title and subtitle for Newton's interpolation method."""
    title = Text("Polynomial Interpolation", color=BLUE).to_edge(UP).scale(1)
    subtitle = Text("Newton's Method", color=GREEN).next_to(title, DOWN, buff=0.5).scale(0.8)

    # Animation: Show title and subtitle
    scene.play(Write(title))
    scene.wait(1)
    scene.play(Write(subtitle))
    scene.wait(1)

    # Move title up and replace with subtitle
    scene.play(
        title.animate.to_edge(UP).shift(UP * 3),
        subtitle.animate.move_to(title.get_center())
    )
    scene.wait(1)
    
    return title, subtitle

def show_conditions(scene, subtitle):
    """Show the interpolation conditions."""
    condition_text = MathTex(
        r"x_i \neq x_j \quad \text{for } i \neq j", font_size=35
    ).next_to(subtitle, DOWN + RIGHT, buff=0.3).set_color(GREEN)

    equation1 = MathTex(
        r"P_n(x_i) = y_i, \quad \forall i \in \{0,1,\dots,n\}",
        font_size=35
    ).next_to(condition_text, DOWN, buff=0.3).set_color(GREEN)

    scene.play(Write(condition_text))
    scene.wait(1)
    scene.play(Write(equation1))
    scene.wait(2)
    
    return condition_text, equation1

def show_polynomial_form(scene, equation1):
    """Show the polynomial form and explanation."""
    polynomial_text = Text(
        "The polynomial can be written as:", font_size=24, color=WHITE
    ).next_to(equation1, DOWN, buff=0.3)

    polynomial_formula = MathTex(
        r"P_n(x) = \sum_{i=0}^{n} \beta_i \omega_i(x)", font_size=40
    ).next_to(polynomial_text, DOWN, buff=0.3).set_color(YELLOW)

    scene.play(Write(polynomial_text))
    scene.play(Write(polynomial_formula))
    scene.wait(2)
    
    return polynomial_text, polynomial_formula

def show_basis_explanation(scene, polynomial_formula):
    """Show explanation of basis polynomials and coefficients."""
    omega_text = MathTex(
        r"\omega_i(x) \text{ are Newton basis polynomials.}", 
        font_size=30
    ).next_to(polynomial_formula, DOWN, buff=0.2).set_color(RED)

    beta_text = MathTex(
        r"\beta_i \text{ are Newton coefficients, computed using } \\ \text{divided differences.}", 
        font_size=30
    ).next_to(omega_text, DOWN, buff=0.2).set_color(RED)

    scene.play(Write(omega_text))
    scene.play(Write(beta_text))
    scene.wait(3)
    
    return omega_text, beta_text

def show_newton_basis_polynomials(scene, polynomial_formula):
    """Show the Newton basis polynomials with transformations."""
    # Title
    title1 = Text("Newton Basis Polynomials", font_size=24, color=BLUE).to_edge(UP)
    title1.next_to(polynomial_formula, DOWN, buff=0.5)
    scene.play(Write(title1))
    scene.wait(1)

    # Définition des polynômes sous leur forme générale
    omega_general = [
        MathTex(r"\omega_0(x) = 1", font_size=36),
        MathTex(r"\omega_1(x) = (x - x_0)", font_size=36),
        MathTex(r"\omega_2(x) = (x - x_0)(x - x_1)", font_size=36),
    ]

    # Positionnement initial à gauche
    for i, omega in enumerate(omega_general):
        omega.next_to(RIGHT).shift(DOWN * i)

    # Affichage initial des formules générales
    scene.play(*[Write(omega) for omega in omega_general])
    scene.wait(2)

    # Définition des polynômes avec les valeurs spécifiques
    omega_specific = [
        MathTex(r"\omega_0(x) = 1", font_size=36),
        MathTex(r"\omega_1(x) = (x - 1)", font_size=36),
        MathTex(r"\omega_2(x) = (x - 1)(x - 2)", font_size=36),
    ]

    # Positionnement pour qu'ils remplacent les formules générales
    for i, omega in enumerate(omega_specific):
        omega.next_to(RIGHT).shift(DOWN * i)

    # Transformation progressive
    scene.play(*[Transform(omega_general[i], omega_specific[i]) for i in range(len(omega_general))])
    scene.wait(3)

    scene.play(FadeOut(*omega_general))
    scene.play(FadeOut(title1))
    scene.wait(1)
    
    return omega_general

def show_newton_polynomial_decomposition(scene, polynomial_formula):
    """Show the Newton polynomial decomposition."""
    # Title
    title2 = Text("Newton Polynomial Decomposition", font_size=25, color=RED).to_edge(UP)
    title2.next_to(polynomial_formula, DOWN, buff=0.5)
    scene.play(Write(title2))
    scene.wait(1)

    newton_poly = MathTex(
        r"P_n(x) = \beta_0 + \beta_1 \omega_1(x)  + \beta_2 \omega_2(x)  \\",
        font_size=40
    ).set_color(BLUE)
    newton_poly.next_to(title2, DOWN, buff=0.5)
    scene.play(Write(newton_poly))
    scene.wait(2)

    newton_poly_val = MathTex(
        r"P_n(x) = \beta_0 + \beta_1 (x - 1)  + \beta_2 (x - 1)(x - 2)  \\",
        font_size=30
    ).set_color(BLUE)
    newton_poly_val.next_to(title2, DOWN, buff=0.5)

    # Transformation progressive
    scene.play(Transform(newton_poly, newton_poly_val))
    scene.wait(3)
    
    return title2, newton_poly

def show_coefficient_explanation(scene, newton_poly):
    """Show explanation of coefficients."""
    # Explanation of coefficients
    explanation_text = Text("Where", font_size=25, color=WHITE)
    explanation_text.next_to(newton_poly, DOWN, buff=0.5)
    scene.play(Write(explanation_text))

    beta_0 = MathTex(r"\beta_0 = y_0", font_size=40).set_color(GREEN)
    beta_1_n = MathTex(
        r"\beta_1, \beta_2, \dots, \beta_n \quad \\ \text{are determined using divided differences.}", 
        font_size=30
    ).set_color(GREEN)
    beta_0.next_to(explanation_text, DOWN, buff=0.3)
    beta_1_n.next_to(beta_0, DOWN, buff=0.3)

    scene.play(Write(beta_0))
    scene.wait(1)
    scene.play(Write(beta_1_n))
    scene.wait(3)
    
    return explanation_text, beta_0, beta_1_n

def show_divided_differences_method(scene, newton_poly, polynomial_formula):
    """Show the method of divided differences."""
    # Faire monter newton_poly_val sous polynomial_formula
    scene.play(newton_poly.animate.next_to(polynomial_formula, DOWN, buff=0.5))
    scene.wait(2)

    # Title
    title3 = Text("Method of Divided Differences", font_size=25, color=GREEN).to_edge(UP)
    title3.next_to(newton_poly, DOWN , buff=0.5)
    scene.play(Write(title3))
    scene.wait(1)

    # General formula for divided differences
    general_formula = MathTex(
        r"f[x_i, x_{i+1}, ..., x_{i+k}] =  \frac{f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]}{x_{i+k} - x_i}", 
        font_size=28
    ).set_color(YELLOW)
    general_formula.next_to(title3, DOWN, buff=0.9)
    scene.play(Write(general_formula))
    scene.wait(3)

    scene.play(FadeOut(general_formula))
    
    return title3

def show_beta_formulas(scene, title3):
    """Show the beta formulas for divided differences."""
    # Beta formulas
    beta_formulas = [
        MathTex(r"\beta_0 = y_0", font_size=38).set_color(GREEN),
        MathTex(r"\beta_1 = \frac{y_1 - y_0}{x_1 - x_0}", font_size=38).set_color(GREEN),
        MathTex(r"\beta_2 = \frac{\frac{y_2 - y_1}{x_2 - x_1} - \frac{y_1 - y_0}{x_1 - x_0}}{x_2 - x_0}", font_size=38).set_color(GREEN)
    ]

    # Positioning beta formulas
    for i, beta in enumerate(beta_formulas):
        beta.next_to(title3, DOWN, buff=0.5 + i * 1)
        scene.play(Write(beta))
        scene.wait(1)
        
    return beta_formulas

def compute_divided_differences(jours, prix):
    """Compute divided differences for the given data points."""
    # Function to compute divided differences
    def differences_divisees(x, y):
        n = len(x)
        table = np.zeros((n, n))
        table[:, 0] = y  # First column = y_i
        
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x[i+j] - x[i])
        
        return table, table[0, :]

    # Compute divided differences
    table_diff, beta_coeffs = differences_divisees(jours, prix)
    
    return table_diff, beta_coeffs

def show_beta_values(scene, beta_formulas, jours, prix, beta_coeffs):
    """Show the beta values with specific numbers and computed results."""
    # Beta values with specific numbers
    beta_values = [
        MathTex(r"\beta_0 = 98", font_size=38),
        MathTex(r"\beta_1 = \frac{105 - 98}{2 - 1}", font_size=38),
        MathTex(r"\beta_2 = \frac{\frac{95 - 105}{3 - 2} - \frac{105 - 98}{2 - 1}}{3 - 1}", font_size=38)
    ]

    # Replacing general formulas with specific values in place
    for i in range(len(beta_formulas)):
        beta_values[i].move_to(beta_formulas[i].get_center())  # Keep position unchanged
        scene.play(Transform(beta_formulas[i], beta_values[i]))
        scene.wait(1)

    # Beta values with computed results
    beta_calculated = [
        MathTex(fr"\beta_0 = {beta_coeffs[0]}", font_size=38),
        MathTex(fr"\beta_1 = {beta_coeffs[1]}", font_size=38),
        MathTex(fr"\beta_2 = {beta_coeffs[2]:.1f}", font_size=38)
    ]

    # Replacing specific values with computed results in place
    for i in range(len(beta_values)):
        beta_calculated[i].move_to(beta_formulas[i].get_center())  # Keep position unchanged
        scene.play(Transform(beta_formulas[i], beta_calculated[i]))
        scene.wait(1)

    scene.wait(2)
    
    return beta_calculated

def show_newton_polynomial(scene, newton_poly, polynomial_formula, beta_values, jours):
    """Show the Newton polynomial with calculated coefficients."""
    # Création d'une nouvelle équation avec les valeurs mises à jour
    newton_poly_final = MathTex(
        fr"P_n(x) = {beta_values[0]} + {beta_values[1]} (x - {jours[0]}) + {beta_values[2]} (x - {jours[0]}) (x - {jours[1]})",
        font_size=30
    ).set_color(BLUE)
    newton_poly_final.next_to(polynomial_formula, DOWN, buff=0.5)

    # Transformation progressive
    scene.play(Transform(newton_poly, newton_poly_final))
    scene.wait(3)
    
    return newton_poly

def show_expanded_polynomial(scene, newton_poly, polynomial_formula, beta_values, jours):
    """Show the expanded polynomial."""
    # Développement du polynôme de Newton
    x = sp.symbols('x')
    # Calcul du polynôme de Newton
    P_n = beta_values[0] + beta_values[1] * (x - jours[0]) + beta_values[2] * (x - jours[0]) * (x - jours[1])
    P_n_simplified = sp.expand(P_n)  # Développement du polynôme

    # Affichage du polynôme développé dans Manim
    pn_equation = MathTex(f"P_n(x) = {sp.latex(P_n_simplified)}", font_size=40)
    pn_equation.set_color(GREEN)

    # Placer P_n(x) à droite de la scène
    pn_equation.next_to(polynomial_formula, DOWN, buff=0.5)

    # Animation de disparition et transition vers pn_equation
    scene.play(Transform(newton_poly, pn_equation))  # Transition entre les équations
    scene.wait(3)

    scene.play(FadeOut(polynomial_formula))

    # Afficher le polynôme final de Newton
    scene.play(Write(pn_equation))
    scene.wait(2)
    
    return pn_equation

def create_axes_and_points(scene, jours, prix):
    """Create axes and plot data points."""
    # Create axes
    axes = Axes(
        x_range=[0, 4, 1],
        y_range=[90, 110, 5],
        axis_config={"color": WHITE},
        x_length=5,
        y_length=4,
    ).to_edge(LEFT)
    
    # Create axis labels
    labels = axes.get_axis_labels(x_label="Jours", y_label=r"Prix (\$)")
    
    # Display axes and labels
    scene.play(Create(axes), Write(labels))
    scene.wait(1)
    
    # Add tick values for X axis
    x_ticks = VGroup(*[Text(str(i), font_size=24).next_to(axes.c2p(i, 0), DOWN * 0.3) for i in range(1, 4)])
    
    # Add tick values for Y axis
    y_ticks = VGroup(*[Text(str(i), font_size=24).next_to(axes.c2p(0, i), LEFT * 0.3) for i in range(90, 111, 5)])
    
    # Display tick values
    scene.play(Write(x_ticks), Write(y_ticks))
    scene.wait(2)
    
    # Plot data points
    point_dots = VGroup(*[Dot(axes.c2p(j, p), color=YELLOW) 
                        for j, p in zip(jours, prix)])
    point_labels = VGroup(*[Text(f"{p:.1f}$", font_size=24)
                          .next_to(Dot(axes.c2p(j, p)), UP) 
                          for j, p in zip(jours, prix)])
    
    scene.play(Create(point_dots), Write(point_labels))
    scene.wait(2)
    
    return axes, labels, x_ticks, y_ticks, point_dots, point_labels

def plot_newton_curve(scene, axes, beta_values, jours):
    """Plot the Newton interpolation curve."""
    # Define the Newton interpolation function
    def newton_interpolation(x_val):
        result = beta_values[0]
        for i in range(1, len(beta_values)):
            term = beta_values[i]
            for j in range(i):
                term *= (x_val - jours[j])
            result += term
        return result
    
    # Now we can plot the Newton curve
    newton_graph = axes.plot(
        lambda x: newton_interpolation(x),
        x_range=[0.5, 3.5],
        color=BLUE
    )
    
    # Show the interpolation curve
    scene.play(Create(newton_graph))
    scene.wait(2)
    
    return newton_graph

class NewtonInterpolation(Scene):
    def construct(self):
        # Initialize data
        jours = np.array([1, 2, 3], dtype=float)
        prix = np.array([98, 105, 95], dtype=float)
        
        # Create axes and plot points first
        axes, labels, x_ticks, y_ticks, point_dots, point_labels = create_axes_and_points(self, jours, prix)
        
        # 1. Show title and subtitle
        title, subtitle = show_title_and_subtitle(self)
        
        # 2. Show conditions
        condition_text, equation1 = show_conditions(self, subtitle)
        
        # 3. Show polynomial form
        polynomial_text, polynomial_formula = show_polynomial_form(self, equation1)
        
        # 4. Show basis explanation
        omega_text, beta_text = show_basis_explanation(self, polynomial_formula)
        
        # 5. Fade out previous elements
        self.play(
            FadeOut(condition_text, equation1, omega_text, beta_text, polynomial_text)
        )
        self.wait(1)
        
        # 6. Move polynomial formula up
        self.play(polynomial_formula.animate.to_edge(UP * 3))
        
        # 7. Show Newton basis polynomials
        omega_general = show_newton_basis_polynomials(self, polynomial_formula)
        
        # 8. Show Newton polynomial decomposition
        title2, newton_poly = show_newton_polynomial_decomposition(self, polynomial_formula)
        
        # 9. Show coefficient explanation
        explanation_text, beta_0, beta_1_n = show_coefficient_explanation(self, newton_poly)
        
        # 10. Fade out previous elements
        self.play(FadeOut(title2, explanation_text, beta_0, beta_1_n))
        self.wait(1)
        
        # 11. Show divided differences method
        title3 = show_divided_differences_method(self, newton_poly, polynomial_formula)
        
        # 12. Show beta formulas
        beta_formulas = show_beta_formulas(self, title3)
        
        # 13. Compute divided differences
        table_diff, beta_coeffs = compute_divided_differences(jours, prix)
        
        # 14. Show beta values
        beta_calculated = show_beta_values(self, beta_formulas, jours, prix, beta_coeffs)
        
        # 15. Show Newton polynomial
        newton_poly = show_newton_polynomial(self, newton_poly, polynomial_formula, beta_coeffs, jours)
        
        # 16. Fade out previous elements
        self.play(FadeOut(title3, *beta_formulas))
        self.wait(1)
        
        # 17. Show expanded polynomial
        pn_equation = show_expanded_polynomial(self, newton_poly, polynomial_formula, beta_coeffs, jours)
        
        # 18. Plot Newton curve at the end
        newton_graph = plot_newton_curve(self, axes, beta_coeffs, jours)
        
        # 19. Clean up scene
        self.play(FadeOut(point_dots, point_labels, newton_graph))
        self.wait(2)
        