from manim import *
import numpy as np
import random
from math import *
config.media_dir="."

class FluctuatingFunction(Scene):
    def construct(self):
        # Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": WHITE}
        )
        
        # Define function
        def fluctuating_function(x):
            return np.sin(2 * np.pi * x) * np.exp(-0.5 * x**2)
        
        # Graph
        graph = axes.plot(fluctuating_function, color=BLUE)
        
        # Animation effect
        glow = axes.plot(lambda x: fluctuating_function(x) * 1.1, color=YELLOW, stroke_width=6)
        
        # Animate
        self.play(Create(axes))
        self.play(Create(graph))
        self.play(Indicate(graph, scale_factor=1.1, color=RED))
        self.play(ShowPassingFlash(glow, time_width=0.5))
        self.wait(2)

class EstimationFunction(Scene):
    def construct(self):
        # Axes
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": WHITE}
        )
        
        # Define estimated function (approximating a trend rather than exact values)
        def estimated_function(x):
            return 0.5 * np.sin(2 * np.pi * x) + 0.3 * x
        
        # Scatter points (representing data with noise)
        scatter_dots = VGroup(*[
            Dot(axes.c2p(x, estimated_function(x) + np.random.uniform(-0.2, 0.2)), color=WHITE)
            for x in np.linspace(-2.5, 2.5, 10)
        ])
        
        # Smooth estimation curve
        estimated_graph = axes.plot(estimated_function, color=GREEN, stroke_width=4)
        
        # Confidence band (simulating uncertainty)
        upper_bound = axes.plot(lambda x: estimated_function(x) + 0.3, color=BLUE, stroke_width=2)
        lower_bound = axes.plot(lambda x: estimated_function(x) - 0.3, color=BLUE, stroke_width=2)
        
        # Animation
        self.play(Create(axes))
        self.play(Create(scatter_dots))
        self.play(Create(estimated_graph))
        self.play(Create(upper_bound), Create(lower_bound))
        self.wait(2)

class HousePricesTable(Scene):
    def construct(self):
        # Define table data
        data = [
            ["ID", "Surface (m²)", "Rooms", "Location", "Price (€)"],
            ["1", "50", "2", "Suburban", "120,000"],
            ["2", "75", "3", "Urban", "200,000"],
            ["3", "100", "4", "Rural", "150,000"],
            ["4", "60", "2", "Urban", "180,000"],
            ["5", "90", "3", "Suburban", "210,000"],
            ["6", "110", "5", "Urban", "350,000"],
            ["7", "45", "1", "Rural", "100,000"],
            ["8", "80", "3", "Suburban", "190,000"],
            ["9", "120", "5", "Urban", "400,000"],
            ["10", "95", "4", "Suburban", "250,000"],
            ["11", "70", "2", "Urban", "170,000"],
            ["12", "85", "3", "Rural", "140,000"],
            ["13", "130", "6", "Urban", "450,000"],
            ["14", "55", "2", "Suburban", "130,000"],
            ["15", "140", "7", "Urban", "500,000"],
            ["16", "75", "3", "Suburban", "200,000"],
            ["17", "50", "2", "Rural", "110,000"],
            ["18", "105", "4", "Urban", "320,000"],
            ["19", "125", "5", "Suburban", "370,000"],
            ["20", "60", "2", "Rural", "115,000"],
        ]
        
        # Create table
        table = Table(
            data,
            include_outer_lines=True
        )
        
        # Scale and position table
        table.scale(0.6)
        table.to_edge(UP)
        
        # Animate table creation
        self.play(Create(table))
        self.wait(2)

class DataDrivenApproach(Scene):
    def construct(self):
        # Step 1: Introduce data (binary tree structure)
        node_scale = 1.5
        depth = 3  # Increased depth of tree
        
        nodes = []
        edges = []
        labels = []  # To store labels for the nodes
        
        # Create nodes and edges dynamically for a deeper tree
        positions = {
            0: ORIGIN + UP * 3,
            1: LEFT * 3 + UP,
            2: RIGHT * 3 + UP,
            3: LEFT * 5 + DOWN,
            4: LEFT * 1 + DOWN,
            5: RIGHT * 1 + DOWN,
            6: RIGHT * 5 + DOWN,
            7: LEFT * 6 + DOWN * 3,
            8: LEFT * 4 + DOWN * 3,
            9: LEFT * 2 + DOWN * 3,
            10: RIGHT * 2 + DOWN * 3,
            11: RIGHT * 4 + DOWN * 3,
            12: RIGHT * 6 + DOWN * 3,
        }
        house_prices = [500000, 400000, 450000, 350000, 420000, 430000, 470000, 300000, 380000, 410000, 460000, 490000, 520000]

        for i in positions:
            node = Dot(positions[i], color=YELLOW).scale(node_scale)
            nodes.append(node)
            if i > 0:
                parent_index = (i - 1) // 2
                edge = Line(nodes[parent_index].get_center(), node.get_center(), color=WHITE)
                edges.append(edge)
            # Add label for house price
            label = Text(str(house_prices[i]), color=WHITE).scale(0.5)
            label.move_to(positions[i] + UP * 0.3)  # Positioning label slightly above the node
            labels.append(label)

        nodes = VGroup(*nodes)
        edges = VGroup(*edges)
        labels = VGroup(*labels)

        self.play(FadeIn(nodes), Create(edges), run_time=3)
        self.play(FadeIn(labels), run_time=1)
        self.wait(1)

        # Step 2: Highlight analysis (wave effect over tree, slower animations)
        for node in nodes:
            self.play(Indicate(node, scale_factor=1.2), run_time=0.6)
        
        # Step 3: Solution found (glowing circle at root)
        lightbulb_alt = Circle(radius=0.7, color=YELLOW, fill_opacity=1)
        lightbulb_alt.move_to(nodes[0].get_center())
        self.play(FadeIn(lightbulb_alt), Indicate(lightbulb_alt, scale_factor=1.3, run_time=1.5))
        
        # End scene
        self.wait(2)



class PricesAndPriceRelationship(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": WHITE},
            tips=False
        ).scale(0.8)

        # Title at the top
        title = Text("How Can Surface of a House Affect Its Price?", font_size=32, color=YELLOW)
        title.to_edge(UP)

        # Labels
        x_label = Text("Surface Area (m²)", font_size=28).next_to(axes.x_axis, DOWN)
        y_label = Text("Price ($)", font_size=28).next_to(axes.y_axis, LEFT)

        # Question mark at the center
        question_mark = Text("?", font_size=60, color=RED)
        question_mark.move_to(axes.c2p(5, 5))  # Position it in the center

        # Animate elements
        self.play(Write(title))
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.play(FadeIn(question_mark, scale=1.5))

        # Hold for a moment before ending
        self.wait(2)

class PeopleTalkingAboutPrices(Scene):
    def construct(self):
        # Create four "people" (circles)
        person_a = Circle(radius=0.6, color=BLUE).shift(LEFT * 4)  # Asking
        person_b = Circle(radius=0.6, color=GREEN).shift(LEFT * 1.5)  # Answering 1
        person_c = Circle(radius=0.6, color=RED).shift(RIGHT * 1.5)  # Answering 2
        person_d = Circle(radius=0.6, color=ORANGE).shift(RIGHT * 4)  # Answering 3

        # Create text bubbles
        question = Text("How much for 200 m² house?").scale(0.5).next_to(person_a, UP)
        answer1 = Text("I bought mine for 140k $, but it's 220 m².").scale(0.4).next_to(person_b, UP)
        answer2 = Text("Mine costs more, 195k $, but it's 250 m².").scale(0.4).next_to(person_c, UP)
        answer3 = Text("Sorry, I didn't catch that.").scale(0.4).next_to(person_d, UP)

        # Animation sequence
        self.play(Create(person_a), Create(person_b), Create(person_c), Create(person_d))
        self.play(Write(question))
        self.play(FadeOut(question))  # Remove second answer

        # Person A asks
        self.play(person_a.animate.shift(UP * 0.3), run_time=0.5)
        self.play(person_a.animate.shift(DOWN * 0.3), run_time=0.5)

        # Each person responds, removing previous answer before the next appears
        self.wait(0.5)
        self.play(person_b.animate.shift(UP * 0.3), Write(answer1), run_time=0.7)
        self.play(person_b.animate.shift(DOWN * 0.3), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(answer1))  # Remove first answer

        self.wait(0.5)
        self.play(person_c.animate.shift(UP * 0.3), Write(answer2), run_time=0.7)
        self.play(person_c.animate.shift(DOWN * 0.3), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(answer2))  # Remove second answer

        self.wait(0.5)
        self.play(person_d.animate.shift(UP * 0.3), Write(answer3), run_time=0.7)
        self.play(person_d.animate.shift(DOWN * 0.3), run_time=0.5)

        # Hold before ending
        self.wait(2)



class NeedForStructuredData(Scene):
    def construct(self):
        # Title
        title = Text("We Need a More Structured Way to Process House Prices", font_size=32, color=YELLOW)
        title.to_edge(UP)

        # Scattered house price data (random positioning)
        data1 = Text("120k $, 85m², 3 rooms", font_size=28, color=RED).shift(LEFT * 4 + UP * 1.5)
        data2 = Text("230k $, 140m², 5 rooms", font_size=28, color=BLUE).shift(LEFT * 2 + DOWN * 1.5)
        data3 = Text("95k $, 65m², 2 rooms", font_size=28, color=GREEN).shift(RIGHT * 3 + UP * 1)
        data4 = Text("310k $, 180m², 6 rooms", font_size=28, color=ORANGE).shift(RIGHT * 1.5 + DOWN * 2)

        scattered_data = VGroup(data1, data2, data3, data4)
        scattered_label = Text("Unorganized data, hard to compare!", font_size=20, color=GRAY).next_to(scattered_data, DOWN)

        # Confused person (circle with question marks)
        thinker = Circle(radius=0.6, color=WHITE).shift(DOWN * 3.5)
        confusion_marks = Text("???", font_size=48, color=RED).move_to(thinker)
        think_label = Text("How do we compare prices?", font_size=20, color=GRAY).next_to(thinker, DOWN)

        # Structured Data (grouped prices by characteristics)
        structured_data = VGroup(
            Text("Small Houses: 95k $ (65m²) - 120k $ (85m²)", font_size=28, color=GREEN).shift(UP * 1.5),
            Text("Medium Houses: 230k $ (140m²)", font_size=28, color=BLUE),
            Text("Large Houses: 310k $ (180m²)", font_size=28, color=ORANGE).shift(DOWN * 1.5),
        )

        structured_label = Text("Now the data is clear and easy to compare!", font_size=20, color=GREEN).next_to(structured_data, DOWN)

        # Animations
        self.play(Write(title))
        self.play(FadeIn(scattered_data), Write(scattered_label))
        self.wait(1)

        self.play(FadeIn(thinker), Write(confusion_marks), Write(think_label))
        self.wait(1)

        self.play(FadeOut(confusion_marks), FadeOut(think_label), FadeOut(scattered_label))
        self.play(Transform(scattered_data, structured_data))
        self.play(Write(structured_label))

        self.wait(2)

        
class HousePriceLinearRelationExplanition(Scene):
    def construct(self):
        # Title
        title = Text("Surface Area Affects House Prices", font_size=32, color=YELLOW)
        title.to_edge(UP)

        # Description text
        description = Text(
            "Larger houses tend to have higher prices. Let's see this relationship visually!",
            font_size=24, color=WHITE
        )
        description.next_to(title, DOWN, buff=0.3)  # Positioning below the title

        # Create axes
        axes = Axes(
            x_range=[0, 400, 50],  # Price ($)
            y_range=[0, 250, 50],  # Surface Area (m²)
            axis_config={"color": WHITE},
            tips=False
        ).scale(0.9)

        # Labels for axes
        x_label = Text("Price ($)", font_size=20).next_to(axes.x_axis, DOWN)
        y_label = Text("Area (m²)", font_size=20).next_to(axes.y_axis, LEFT)

        # Initial line (flat line)
        initial_line = axes.plot(lambda x: 50, color=BLUE)

        # Different slopes (animated)
        slope1 = axes.plot(lambda x: 0.2 * x, color=BLUE)
        slope2 = axes.plot(lambda x: 0.4 * x, color=BLUE)
        slope3 = axes.plot(lambda x: 0.6 * x, color=BLUE)

        # Adjusted equation position and size
        equation = MathTex("y = mx + b", font_size=36, color=GREEN)
        equation.next_to(axes, DOWN, buff=0.7)  # Places it just below the graph

        # Animations
        self.play(Write(title))
        self.play(FadeIn(description))  # Smooth fade-in for description
        self.wait(1)
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # Animate the changing slope
        self.play(Create(initial_line))
        self.wait(0.5)
        self.play(Transform(initial_line, slope1), run_time=2)
        self.play(Transform(initial_line, slope2), run_time=2)
        self.play(Transform(initial_line, slope3), run_time=2)

        # Show equation
        self.wait(1)
        self.play(Write(equation))

        self.wait(2)


class VariablesExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Understanding the Equation: y = mx + b", font_size=32, color=YELLOW)
        title.to_edge(UP)

        # Equation on the right side
        equation = MathTex("y = mx + b", font_size=48)
        equation.to_edge(RIGHT, buff=1)

        # Descriptions for each variable, positioned on the left side
        y_description = Text("y = Price ($)", font_size=28, color=WHITE).to_edge(LEFT, buff=1)
        m_description = Text("m = Slope (Price increase per m²)", font_size=20, color=WHITE).next_to(y_description, DOWN, buff=0.6)
        x_description = Text("x = Area (m²)", font_size=28, color=WHITE).next_to(m_description, DOWN, buff=0.6)
        b_description = Text("b = Baseline", font_size=28, color=WHITE).next_to(x_description, DOWN, buff=0.6)

        # Create animations
        self.play(Write(title))
        self.wait(1)

        # Show the equation and explanations in a clean way
        self.play(Write(equation))
        self.wait(1)

        # Introduce each description one by one on the left side
        self.play(Write(y_description))
        self.wait(0.5)
        self.play(Write(m_description))
        self.wait(0.5)
        self.play(Write(x_description))
        self.wait(0.5)
        self.play(Write(b_description))

        # Add simple animation for interactivity: highlighting the equation
        self.play(equation.animate.set_color(YELLOW), run_time=1)
        self.wait(0.5)

        # Simple movement animation to focus on equation and left text
        self.play(
            equation.animate.shift(LEFT * 2),
            y_description.animate.shift(RIGHT * 2),
            m_description.animate.shift(RIGHT * 2),
            x_description.animate.shift(RIGHT * 2),
            b_description.animate.shift(RIGHT * 2),
            run_time=1
        )

        self.wait(2)



class Example(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(
            x_range=[0, 10],  # Surface area range (0 to 10)
            y_range=[0, 10],  # Price range (0 to 100)
            axis_config={"color": BLUE},
        )

        # Create the labels
        x_label = axes.get_x_axis_label("Surface Area (m²)")
        y_label = axes.get_y_axis_label("Price (\\$)")

        # Create the linear line y = 10x (representing house price depending on surface area)
        line = axes.plot(lambda x: x, color=RED)

        # Add a title to the plot
        title = Text("Simple Example of House Price and Surface", font_size=24).to_edge(UP)

        # Hardcoded points and their labels
        points = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]

        # Create the points and labels
        point_objects = []
        for x_val, y_val in points:
            point = Dot(axes.c2p(x_val, y_val), color=YELLOW)
            label = Text(f"({x_val*100}m2, {y_val*100000}$)", font_size=18).next_to(point, UP)
            point_objects.append((point, label))

        # Add everything to the scene
        self.play(Create(axes), Write(title), Write(x_label), Write(y_label))
        self.play(Create(line))

        # Add points and labels
        for point, label in point_objects:
            self.play(FadeIn(point), Write(label))

        # Wait for a while before ending the animation
        self.wait(2)

class MissingEquation(Scene):
    def construct(self):
        # Title text
        title = Text("Our equation is missing something!", font_size=36)
        title.to_edge(UP)

        # Create the initial equation with blanks
        equation = Tex("y = \\_ x + \\_")
        equation.set_font_size(48)
        equation.move_to(ORIGIN)

        # Add title and equation to the scene
        self.play(Write(title))
        self.play(Write(equation))
        
        # Create the parts that represent the missing "m" and "b"
        m_label = Text("m", font_size=48, color=YELLOW)
        b_label = Text("b", font_size=48, color=YELLOW)

        # Animate the missing parts "m" and "b" appearing in the equation
        self.play(
            m_label.animate.move_to(LEFT * 1.5 + UP * 0.75),
            b_label.animate.move_to(RIGHT * 1.5 + UP * 0.75)
        )

        # Transform the equation into the completed one: y = mx + b
        final_equation = Tex("y = ? . x + ?")
        final_equation.set_font_size(48)
        final_equation.move_to(ORIGIN)

        # Transition to the final equation
        self.play(Transform(equation, final_equation))
        
        # Add a final statement
        conclusion = Text("how to get this values ?", font_size=24)
        conclusion.next_to(final_equation, DOWN)

        self.play(Write(conclusion))

        # Wait before ending the scene
        self.wait(2)

class LinearRegressionScene(Scene):
    def construct(self):
        # Title
        title = Text("Linear function using our data", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create axes for the graph
        axes = Axes(
            x_range=[0, 6],
            y_range=[0, 15],
            axis_config={"color": BLUE},
        )
        axes_labels = axes.get_axis_labels(x_label="Surface Area (m²)", y_label=r"Price (\$)")
        
        # Display axes
        self.play(Create(axes), Write(axes_labels))
        
        # Generate more diverse random scatter points with wider distribution
        scatter_points = []
        for _ in range(50):  # Random 50 points for more variety
            x = random.uniform(1, 5)  # Random x-coordinate between 1 and 5
            y = 2 * x + 5 + random.uniform(-3, 3)  # Linear relation with more random deviation
            
            scatter_points.append(Dot(axes.c2p(x, y), color=RED))
        
        # Animate scatter points appearing
        self.play(LaggedStart(*[GrowFromCenter(point) for point in scatter_points], lag_ratio=0.1))
        
        # Create the linear regression line (y = 2x + 5)
        line = axes.plot(lambda x: 2 * x + 5, color=GREEN)
        
        # Animate the linear regression line being drawn
        self.play(Create(line))
        
        # Final annotation and fade out the title
        final_text = Text("Draws a line that passes by the most number of points", font_size=24).next_to(line, DOWN)
        self.play(Write(final_text))
        
        # Wait a bit
        self.wait(2)
        
        # Fade out everything
        self.play(FadeOut(title), FadeOut(axes), FadeOut(axes_labels), FadeOut(final_text), FadeOut(line), FadeOut(*scatter_points))


from manim import *
import numpy as np
import random

class LeastSquaresWithLabels(Scene):
    def construct(self):
        # Title
        title = Text("Least Squares Method Explained", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Create axes
        axes = Axes(
            x_range=[0, 6],
            y_range=[0, 15],
            axis_config={"color": BLUE},
        )
        axes_labels = axes.get_axis_labels(x_label="Surface Area (m²)", y_label=r"Price (\$)")
        self.play(Create(axes), Write(axes_labels))

        # Generate random scattered points with a linear trend
        scatter_points = []
        x_values = np.linspace(1, 5, 10)
        y_values = [2*x + 5 + random.uniform(-2, 2) for x in x_values]  # Random variation
        for x, y in zip(x_values, y_values):
            scatter_points.append(Dot(axes.c2p(x, y), color=RED))

        self.play(LaggedStart(*[GrowFromCenter(point) for point in scatter_points], lag_ratio=0.1))

        # Left-side labels (stacked vertically)
        points_label = Text("Observed Data Points", font_size=22, color=RED)
        points_label.to_edge(LEFT).shift(UP * 1.5)
        self.play(Write(points_label))
        self.wait(2)

        # Show a "wrong" line first
        wrong_line = axes.plot(lambda x: 3*x + 2, color=YELLOW)
        wrong_label = Text("Is this the best line?", font_size=24).next_to(wrong_line, UP)
        self.play(Create(wrong_line), Write(wrong_label))
        self.wait(2)

        # Right-side label (stacked)
        wrong_line_label = Text("Guess: Not the best fit", font_size=22, color=YELLOW)
        wrong_line_label.to_edge(RIGHT).shift(UP * 1.5)
        self.play(Write(wrong_line_label))
        self.wait(2)

        # Show errors (distances from points to the line)
        errors = []
        for x, y in zip(x_values, y_values):
            y_pred = 3*x + 2  # y-value on the wrong line
            error_line = DashedLine(axes.c2p(x, y), axes.c2p(x, y_pred), color=WHITE)
            errors.append(error_line)

        self.play(*[Create(err) for err in errors])

        # Next right-side label (stacked below)
        error_text = Text("Errors = Distance from line", font_size=20, color=WHITE)
        error_text.to_edge(RIGHT).shift(UP * 0.5)
        self.play(Write(error_text))
        self.wait(2)

        # Explain Squaring the Errors
        squares = []
        for x, y in zip(x_values, y_values):
            y_pred = 3*x + 2
            square = Square(side_length=0.4, color=ORANGE).move_to(axes.c2p(x, (y + y_pred)/2))
            squares.append(square)

        self.play(*[Create(sq) for sq in squares])

        # Next left-side label (stacked below)
        squared_error_label = Text("We square errors\nto give larger mistakes\n more weight", font_size=20, color=ORANGE)
        squared_error_label.to_edge(LEFT).shift(UP * 0.5)
        self.play(Write(squared_error_label))
        self.wait(2)

        # Remove wrong line and errors
        self.play(FadeOut(wrong_line), FadeOut(wrong_label), FadeOut(wrong_line_label))
        self.play(FadeOut(*errors), FadeOut(error_text))
        self.play(FadeOut(*squares), FadeOut(squared_error_label))

        # Show the correct Least Squares Regression Line
        correct_line = axes.plot(lambda x: 2*x + 5, color=GREEN)
        correct_label = Text("Best-fitting line using Least Squares!", font_size=24).next_to(correct_line, UP)
        self.play(Create(correct_line), Write(correct_label))

        # Next right-side label (stacked below)
        best_fit_label = Text("Minimizes total squared errors", font_size=22, color=GREEN)
        best_fit_label.to_edge(RIGHT).shift(DOWN * 0.5)
        self.play(Write(best_fit_label))
        self.wait(2)

        # Final equation display
        equation = MathTex("y = 2x + 5", color=GREEN).to_edge(DOWN)
        self.play(Write(equation))
        self.wait(3)

        # Fade out everything
        self.play(FadeOut(title), FadeOut(axes), FadeOut(axes_labels), FadeOut(*scatter_points))
        self.play(FadeOut(correct_line), FadeOut(correct_label), FadeOut(best_fit_label), FadeOut(equation), FadeOut(points_label))

class LeastSquaresMatrixFormDetailedExplanation(Scene):
    def construct(self):
        # Title
        title = Text("Least Squares Method (House Price Example)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Step 1: Define Matrices
        A_matrix = MathTex(
            r"A = \begin{bmatrix} 1 & S_1 \\ 1 & S_2 \\ \vdots & \vdots \\ 1 & S_n \end{bmatrix}",
            color=BLUE
        ).to_edge(LEFT)

        Lambda_matrix = MathTex(
            r"\Lambda = \begin{bmatrix} \lambda_0 \\ \lambda_1 \end{bmatrix}",
            color=GREEN
        ).next_to(A_matrix, RIGHT, buff=1)

        Y_matrix = MathTex(
            r"Y = \begin{bmatrix} P_1 \\ P_2 \\ \vdots \\ P_n \end{bmatrix}",
            color=RED
        ).next_to(Lambda_matrix, RIGHT, buff=1)

        label_step1 = Text("House surface & prices in a table", font_size=24)
        label_step1.next_to(A_matrix, DOWN, buff=1)

        self.play(Write(A_matrix), Write(Lambda_matrix), Write(Y_matrix))
        self.play(Write(label_step1))
        self.wait(2)

        # Step 2: Define the Least Squares Problem
        least_squares_eq = MathTex(r"A \Lambda \approx Y")
        least_squares_eq.next_to(Y_matrix, RIGHT, buff=1.5)

        label_step2 = Text("predict the price formula", font_size=24)
        label_step2.next_to(least_squares_eq, DOWN, buff=1)

        self.play(Write(least_squares_eq))
        self.play(Write(label_step2))
        self.wait(2)
        self.play(FadeOut(least_squares_eq))
        multiply_eq = MathTex(r"A^T A \Lambda = A^T Y")
        multiply_eq.next_to(least_squares_eq, DOWN, buff=1.5)

        label_step3 = Text("Fix overlap by multiplying A^t to square all matrices", font_size=24)
        label_step3.next_to(multiply_eq, DOWN, buff=1)

        self.play(Write(multiply_eq))
        self.play(Write(label_step3))
        self.wait(2)

        # Step 4: Solve for Lambda
        solve_lambda = MathTex(r"\Lambda = (A^T A)^{-1} A^T Y", color=GREEN)
        solve_lambda.next_to(multiply_eq, DOWN, buff=1.5)

        label_step4 = Text("we got our equations for \n the slop and baseline!", font_size=24)
        label_step4.next_to(solve_lambda, DOWN, buff=1)

        self.play(Write(solve_lambda))
        self.play(Write(label_step4))
        self.wait(2)

        # Fade everything out
        self.play(FadeOut(title, A_matrix, Lambda_matrix, Y_matrix, label_step1,
                          least_squares_eq, label_step2, multiply_eq, label_step3,
                          solve_lambda, label_step4))


class SolvingUsingAverages(Scene):
    def construct(self):
        # Title
        title = Text("Solving Using Averages (House Price Prediction)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Step 1: Introduce the Means (Surface area, Prices, etc.)

        # Mean of Surface Areas (x_bar)
        x_bar = MathTex(r"\bar{x} = \frac{1}{n} \sum_{i=1}^n S_i", color=YELLOW).shift(UP * 2)
        label_x_bar = Text("Mean of Surface Areas", font_size=20).next_to(x_bar, DOWN, buff=0.5)
        text_x_bar = Text("Average of surface areas", font_size=18).next_to(label_x_bar, DOWN, buff=0.5)

        # Mean of Prices (y_bar)
        y_bar = MathTex(r"\bar{y} = \frac{1}{n} \sum_{i=1}^n P_i", color=ORANGE).shift(UP * 2 + RIGHT * 4)
        label_y_bar = Text("Mean of Prices", font_size=20).next_to(y_bar, DOWN, buff=0.5)
        text_y_bar = Text("Average of house prices", font_size=18).next_to(label_y_bar, DOWN, buff=0.5)

        # Display the means and labels step by step
        self.play(FadeIn(x_bar), Write(label_x_bar))
        self.play(Write(text_x_bar))
        self.wait(1)
        self.play(FadeIn(y_bar), Write(label_y_bar))
        self.play(Write(text_y_bar))
        self.wait(2)

        # Step 2: Mean of Surface * Price Products (xy_bar)
        xy_bar = MathTex(r"\bar{xy} = \frac{1}{n} \sum_{i=1}^n S_i P_i", color=BLUE).shift(DOWN * 1.5)
        label_xy_bar = Text("Mean of Surface * Price Products", font_size=20).next_to(xy_bar, DOWN, buff=0.5)
        text_xy_bar = Text("Average of surface area * price", font_size=18).next_to(label_xy_bar, DOWN, buff=0.5)
        
        self.play(FadeIn(xy_bar), Write(label_xy_bar))
        self.play(Write(text_xy_bar))
        self.wait(2)

        # Step 3: Mean of Squared Surface Areas (x_squared_bar) - Adjusted Position
        x_squared_bar = MathTex(r"\bar{x^2} = \frac{1}{n} \sum_{i=1}^n S_i^2", color=PURPLE).shift(UP * 2 + LEFT * 4)
        label_x_squared_bar = Text("Mean of Squared Surface Areas", font_size=20).next_to(x_squared_bar, DOWN, buff=0.5)
        text_x_squared_bar = Text("Average of squared surface areas", font_size=18).next_to(label_x_squared_bar, DOWN, buff=0.5)

        self.play(FadeIn(x_squared_bar), Write(label_x_squared_bar))
        self.play(Write(text_x_squared_bar))
        self.wait(2)

        # Step 4: Transition to λ₁ (Slope) and λ₀ (Intercept) formulas
        
        # Fade out previous elements before lambda_1
        self.play(FadeOut(x_bar, y_bar, xy_bar, x_squared_bar, label_x_bar, label_y_bar, label_xy_bar, label_x_squared_bar, 
                          text_x_bar, text_y_bar, text_xy_bar, text_x_squared_bar))
        
        # Green function (λ₁) now appears centered (default anchor)
        lambda_1 = MathTex(
            r"\lambda_1 = \frac{\bar{xy} - \bar{x} \bar{y}}{\bar{x^2} - \bar{x}^2}", color=GREEN
        )
        
        label_lambda_1 = Text("Slope (λ₁)", font_size=20).next_to(lambda_1, DOWN, buff=0.5)
        text_lambda_1 = Text("Slope of the regression line", font_size=18).next_to(label_lambda_1, DOWN, buff=0.5)

        self.play(FadeIn(lambda_1), Write(label_lambda_1))
        self.play(Write(text_lambda_1))
        self.wait(1)

        # λ₀ (Intercept) appears below λ₁
        lambda_0 = MathTex(
            r"\lambda_0 = \bar{y} - \lambda_1 \bar{x}", color=RED
        ).shift(DOWN * 2)
        
        label_lambda_0 = Text("Intercept (λ₀)", font_size=20).next_to(lambda_0, DOWN, buff=0.5)
        text_lambda_0 = Text("Intercept of the regression line", font_size=18).next_to(label_lambda_0, DOWN, buff=0.5)
        
        self.play(FadeIn(lambda_0), Write(label_lambda_0))
        self.play(Write(text_lambda_0))
        self.wait(2)

        # Step 5: Matrix form of λ₁ and λ₀
        
        matrix_form = MathTex(
            r"\begin{bmatrix} \lambda_1 \\ \lambda_0 \end{bmatrix} = \begin{bmatrix} \frac{\bar{xy} - \bar{x} \bar{y}}{\bar{x^2} - \bar{x^2}} \\ \bar{y} - \lambda_1 \bar{x} \end{bmatrix}", color=WHITE
        ).shift(DOWN * 3)
        label_matrix = Text("Matrix form of the solution", font_size=20).next_to(matrix_form, DOWN, buff=0.5)

        self.play(Transform(lambda_1, matrix_form))
        self.play(Write(label_matrix))
        self.wait(2)

        # Step 6: Final Linear Equation for House Prices
        
        final_eq = MathTex(
            r"P = \lambda_0 + \lambda_1 S", color=YELLOW
        ).shift(DOWN * 6)

        label_final_eq = Text("Final Prediction Equation", font_size=20).next_to(final_eq, DOWN, buff=0.5)

        self.play(Write(final_eq))
        self.play(Write(label_final_eq))
        self.wait(2)

        # Fade everything out
        self.play(FadeOut(title, lambda_1, lambda_0, label_lambda_1, label_lambda_0, final_eq, label_final_eq))

class SolvingUsingAveragesBeta(Scene):
    def construct(self):
        # Title
        title = Text("House Price Prediction Using Averages", font_size=32)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # Step 1: Introduce the Means in a staggered layout
        # First row (x_bar and y_bar)
        x_bar = MathTex(r"\bar{x} = \frac{1}{n} \sum_{i=1}^n S_i", color=YELLOW)
        x_bar_label = Text("Mean Surface Area", font_size=20).next_to(x_bar, DOWN, buff=0.2)
        x_bar_group = VGroup(x_bar, x_bar_label).shift(LEFT*3.5 + UP*1)
        
        y_bar = MathTex(r"\bar{y} = \frac{1}{n} \sum_{i=1}^n P_i", color=ORANGE)
        y_bar_label = Text("Mean Price", font_size=20).next_to(y_bar, DOWN, buff=0.2)
        y_bar_group = VGroup(y_bar, y_bar_label).shift(RIGHT*3.5 + UP*1)
        
        # Second row (x_squared_bar and xy_bar)
        x_squared_bar = MathTex(r"\bar{x^2} = \frac{1}{n} \sum_{i=1}^n S_i^2", color=PURPLE)
        x_squared_label = Text("Mean Squared Area", font_size=20).next_to(x_squared_bar, DOWN, buff=0.2)
        x_squared_group = VGroup(x_squared_bar, x_squared_label).shift(LEFT*3.5 + DOWN*1.2)
        
        xy_bar = MathTex(r"\bar{xy} = \frac{1}{n} \sum_{i=1}^n S_i P_i", color=BLUE)
        xy_bar_label = Text("Mean Area×Price", font_size=20).next_to(xy_bar, DOWN, buff=0.2)
        xy_bar_group = VGroup(xy_bar, xy_bar_label).shift(RIGHT*3.5 + DOWN*1.2)
        
        # Animate in two phases
        self.play(FadeIn(x_bar_group), FadeIn(y_bar_group))
        self.wait(1)
        self.play(FadeIn(x_squared_group), FadeIn(xy_bar_group))
        self.wait(2)
        
        # Clear screen with transform
        self.play(
            FadeOut(x_bar_group, y_bar_group),
            x_squared_group.animate.scale(0.7).to_edge(LEFT, buff=1),
            xy_bar_group.animate.scale(0.7).to_edge(RIGHT, buff=1),
        )
        
        # Step 2: Slope formula
        lambda_1 = MathTex(
            r"\lambda_1 = \frac{\bar{xy} - \bar{x}\bar{y}}{\bar{x^2} - \bar{x}^2}", 
            color=GREEN
        ).shift(UP*0.5)
        
        lambda_1_label = Text("Slope Coefficient", font_size=24).next_to(lambda_1, DOWN, buff=0.3)
        
        self.play(Write(lambda_1), Write(lambda_1_label))
        self.wait(2)
        
        # Step 3: Intercept formula
        lambda_0 = MathTex(
            r"\lambda_0 = \bar{y} - \lambda_1 \bar{x}", 
            color=RED
        ).next_to(lambda_1, UP, buff=1)
        
        lambda_0_label = Text("Base Price (Intercept)", font_size=24).next_to(lambda_0, DOWN, buff=0.3)
        
        self.play(Write(lambda_0), Write(lambda_0_label))
        self.wait(2)
        
        # Step 4: Corrected matrix form
        matrix_form = MathTex(
            r"\begin{bmatrix} \lambda_1 \\ \lambda_0 \end{bmatrix} = " +
            r"\begin{bmatrix} " +
            r"\dfrac{\bar{xy} - \bar{x}\bar{y}}{\bar{x^2} - \bar{x}^2} \\ " +
            r"\bar{y} - \lambda_1 \bar{x} " +
            r"\end{bmatrix}", 
            color=WHITE
        ).scale(0.9).next_to(lambda_1, DOWN*0.75, buff=1.2)
        
        matrix_label = Text("Solution Vector", font_size=22).next_to(matrix_form, DOWN*0.2, buff=0.3)
        
        self.play(Write(matrix_form), Write(matrix_label))
        self.wait(2)
        
        # Step 5: Final equation
        final_eq = MathTex(
            r"P(S) = \lambda_0 + \lambda_1 S", 
            color=YELLOW
        ).next_to(matrix_form, DOWN * 0.25, buff=1.5)
        
        final_label = Text("Price Prediction Formula", font_size=24).next_to(final_eq, DOWN, buff=0.3)
        
        self.play(Write(final_eq), Write(final_label))
        self.wait(3)
        
        # Clean exit
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class EnhancedPolynomialRegression(Scene):
    def construct(self):
        # Setup colors
        BLUE_D = "#1F77B4"
        ORANGE_D = "#FF7F0E"
        GREEN_D = "#2CA02C"
        PURPLE_D = "#9467BD"
        RED_D = "#D62728"
        
        # ========== 1. Linear vs Polynomial Comparison (20 sec) ==========
        title = Text("Linear vs Polynomial Regression", font_size=36)
        self.play(Write(title))
        self.wait(1)
        
        # Create sample data that's clearly non-linear
        x_vals = np.linspace(1, 7, 7)
        y_vals = np.array([2, 3, 5, 9, 15, 23, 33])
        
        axes = Axes(
            x_range=[0, 8, 1],
            y_range=[0, 35, 5],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE}
        ).shift(DOWN*0.5)
        
        data_points = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE_D) 
            for x, y in zip(x_vals, y_vals)
        ])
        
        # Linear regression line
        linear_fit = axes.plot(
            lambda x: 5.2*x - 10.5,
            color=RED_D
        )
        
        # Polynomial regression curve
        poly_fit = axes.plot(
            lambda x: 0.5*x**2 - 0.5*x + 1.5,
            color=GREEN_D
        )
        
        # Animate
        self.play(
            title.animate.to_edge(UP),
            Create(axes),
            LaggedStart(*[GrowFromCenter(dot) for dot in data_points], lag_ratio=0.2)
        )
        self.wait(1)
        
        # Show linear fit
        linear_label = Text("Linear Fit", font_size=24, color=RED_D).next_to(axes, UP, buff=0.1)
        self.play(
            Create(linear_fit),
            Write(linear_label)
        )
        self.wait(2)
        
        # Show polynomial fit
        poly_label = Text("Polynomial Fit", font_size=24, color=GREEN_D).next_to(linear_label, RIGHT, buff=1)
        self.play(
            Create(poly_fit),
            Write(poly_label)
        )
        
        # Explanation text
        explanation = Text(
            "Polynomial fits curved data better\nby adding higher-order terms",
            font_size=24,
            t2c={"curved": GREEN_D}
        ).next_to(axes, DOWN, buff=0.8)
        
        self.play(Write(explanation))
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(data_points),
            FadeOut(linear_fit),
            FadeOut(poly_fit),
            FadeOut(linear_label),
            FadeOut(poly_label),
            FadeOut(explanation)
        )
        
        # ========== 2. Polynomial Model Setup ==========
        equation = MathTex(
            r"P(S) = aS^2 + bS + c",
            substrings_to_isolate=["a", "b", "c", "S", "P"]
        )
        equation.set_color_by_tex("a", GREEN_D)
        equation.set_color_by_tex("b", ORANGE_D)
        equation.set_color_by_tex("c", PURPLE_D)
        equation.set_color_by_tex("S", BLUE_D)
        equation.set_color_by_tex("P", RED_D)
        
        eq_title = Text("Quadratic Price Model", font_size=30).to_edge(UP)
        
        self.play(Write(eq_title))
        self.play(Write(equation))
        self.wait(1)
        self.play(FadeOut(eq_title))
        # ========== 3. Data Points Setup ==========
        # Create initial positions for dots and labels together
        data_groups = VGroup()
        for i, x in enumerate(range(1, 8)):
            dot = Dot(color=BLUE_D).move_to([x-4, (0.5*x**2 - 0.5*x + 1.5)/5-1.5, 0])
            label = MathTex(f"S_{i+1}", color=WHITE, font_size=20).next_to(dot, UP, buff=0.1)
            group = VGroup(dot, label)
            data_groups.add(group)
        
        
        self.wait(1)
        
        # ========== 4. Enhanced Matrix Calculation ==========
        # Step 1: Show X matrix construction
        x_matrix = MathTex(
            r"X = \begin{bmatrix} "
            r"1^2 & 1 & 1 \\ "
            r"2^2 & 2 & 1 \\ "
            r"3^2 & 3 & 1 \\ "
            r"\vdots & \vdots & \vdots \\ "
            r"7^2 & 7 & 1 "
            r"\end{bmatrix}",
            font_size=36
        ).set_color(BLUE_D)
        
        self.play(
            equation.animate.to_edge(UP),
            FadeIn(x_matrix)
        )
        self.wait(2)
        
        # Step 2: Show X^T X calculation
        xtx_matrix = MathTex(
            r"X^T X = \begin{bmatrix} "
            r"\sum S_i^4 & \sum S_i^3 & \sum S_i^2 \\ "
            r"\sum S_i^3 & \sum S_i^2 & \sum S_i \\ "
            r"\sum S_i^2 & \sum S_i & n \\ "
            r"\end{bmatrix}",
            font_size=36
        ).set_color(ORANGE_D)
        
        self.play(
            Transform(x_matrix, xtx_matrix)
        )
        self.wait(2)
        
        # Step 3: Show inversion
        inv_matrix = MathTex(
            r"(X^T X)^{-1} = \text{Inverse of above}",
            font_size=36
        ).set_color(PURPLE_D)
        
        self.play(
            Transform(x_matrix, inv_matrix)
        )
        self.wait(2)
        
        # Step 4: Full solution
        solution = MathTex(
            r"\begin{bmatrix} a \\ b \\ c \end{bmatrix} = ",
            r"(X^T X)^{-1} X^T ",
            r"\begin{bmatrix} P_1 \\ P_2 \\ \vdots \\ P_7 \end{bmatrix}",
            font_size=36
        )
        solution[0].set_color(GREEN_D)
        solution[1].set_color(PURPLE_D)
        solution[2].set_color(RED_D)
        
        self.play(
            FadeOut(x_matrix),
            Write(solution)
        )
        self.wait(3)
        
        # ========== 5. Accurate Final Graph ==========
        self.play(
            LaggedStart(*[FadeIn(group) for group in data_groups], lag_ratio=0.2)
        )
        final_axes = Axes(
            x_range=[0, 8, 1],
            y_range=[0, 35, 5],
            x_length=6,
            y_length=4,
            axis_config={"color": WHITE}
        ).shift(DOWN*1)
        
        # Create target positions for dots and labels
        final_groups = VGroup()
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            dot = Dot(final_axes.c2p(x, y), color=BLUE_D)
            label = MathTex(f"S_{i+1}", color=WHITE, font_size=20).next_to(dot, UP, buff=0.1)
            group = VGroup(dot, label)
            final_groups.add(group)
        
        accurate_curve = final_axes.plot(
            lambda x: 0.5*x**2 - 0.5*x + 1.5,
            color=GREEN_D
        )
        
        final_eq = MathTex(
            r"P(S) = 0.5S^2 - 0.5S + 1.5",
            font_size=36,
            color=YELLOW
        ).next_to(final_axes, UP, buff=0.5)
        
        self.play(
            FadeOut(solution),
            Create(final_axes),
            *[Transform(data_groups[i], final_groups[i]) for i in range(7)],
            Write(final_eq)
        )
        
        self.play(
            Create(accurate_curve, run_time=3)
        )
        
        # Highlight perfect fit
        for group in final_groups:
            self.play(Indicate(group[0], scale_factor=1.3), run_time=0.3)
        
        self.wait(2)
        
        # Final conclusion
        conclusion = Text(
            "Polynomial regression perfectly fits\nthe non-linear house price data!",
            font_size=28,
            color=WHITE
        ).next_to(final_axes, DOWN, buff=0.8)
        
        self.play(Write(conclusion))
        self.wait(3)
        
        # Clean exit
        self.play(*[FadeOut(mob) for mob in self.mobjects])

