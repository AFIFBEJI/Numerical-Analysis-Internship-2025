from manim import *
import numpy as np
import random
from math import *

class RegressionApplications(Scene):
    def construct(self):
        # Color scheme
        BLUE_D = "#1F77B4"
        ORANGE_D = "#FF7F0E"
        GREEN_D = "#2CA02C"
        PURPLE_D = "#9467BD"
        
        # Title
        title = Text("Real-World Applications", font_size=40, color=WHITE)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # ===== LINEAR REGRESSION APPLICATIONS =====
        linear_title = Text("Linear Regression", font_size=32, color=BLUE_D)
        linear_title.move_to(UP*2.5)
        self.play(Write(linear_title))
        
        # Application 1: Sales Prediction
        app1 = VGroup(
            Text("1. Sales Prediction", font_size=28, color=WHITE),
            Text("Predicting sales based on\nadvertising budget", font_size=22, color=GRAY_A)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(linear_title, DOWN, buff=1)
        
        # Create animated sales graph
        sales_axes = Axes(
            x_range=[0, 100, 20], y_range=[0, 200, 50],
            x_length=4, y_length=2.5,
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": [0,50,100]},
            y_axis_config={"numbers_to_include": [0,100,200]}
        ).next_to(app1, DOWN, buff=0.5)
        
        sales_line = sales_axes.plot(lambda x: 1.8*x + 10, color=BLUE_D)
        sales_dots = VGroup(*[
            Dot(sales_axes.c2p(x, 1.8*x + 10 + random.uniform(-15,15)), color=BLUE_D)
            for x in np.linspace(10, 90, 6)
        ])
        
        # Animate with dollar signs
        dollar_signs = VGroup(*[
            Text("$", font_size=24, color=GREEN_D).move_to(
                sales_axes.c2p(x+5, 1.8*x + 20 + random.uniform(0,20)))
            for x in np.linspace(10, 90, 4)
        ])
        
        self.play(FadeIn(app1))
        self.play(
            Create(sales_axes),
            LaggedStart(*[GrowFromCenter(dot) for dot in sales_dots], lag_ratio=0.2),
            Create(sales_line),
            run_time=2
        )
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dollar_signs], lag_ratio=0.3),
            run_time=2
        )
        self.wait(2)
        
        # Application 2: Medical Studies (Modified version without SVG)
        app2 = VGroup(
            Text("2. Medical Studies", font_size=28, color=WHITE),
            Text("Drug dosage vs.\npatient response", font_size=22, color=GRAY_A)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(sales_axes, DOWN, buff=1)
        
        # Create medical chart with capsule shapes made from rectangles and circles
        medical_axes = Axes(
            x_range=[0, 10], y_range=[0, 100, 20],
            x_length=4, y_length=2.5,
            axis_config={"color": WHITE}
        ).next_to(app2, DOWN, buff=0.5)
        
        medical_line = medical_axes.plot(lambda x: 8*x + 5, color=BLUE_D)
        medical_dots = VGroup(*[
            Dot(medical_axes.c2p(x, 8*x + 5 + random.uniform(-10,10)), color=BLUE_D)
            for x in np.linspace(1, 9, 5)
        ])
        
        # Create simple pill shapes using rectangles and semicircles
        pills = VGroup()
        for x in np.linspace(1, 9, 5):
            pill_body = Rectangle(width=0.4, height=0.2, fill_opacity=1, color=BLUE_D)
            pill_left = Circle(radius=0.1, fill_opacity=1, color=BLUE_D).move_to(pill_body.get_left())
            pill_right = Circle(radius=0.1, fill_opacity=1, color=BLUE_D).move_to(pill_body.get_right())
            pill = VGroup(pill_body, pill_left, pill_right)
            pill.move_to(medical_axes.c2p(x, -10))
            pills.add(pill)
        
        self.play(FadeIn(app2))
        self.play(
            Create(medical_axes),
            LaggedStart(*[GrowFromCenter(dot) for dot in medical_dots], lag_ratio=0.2),
            Create(medical_line),
            run_time=2
        )
        self.play(
            LaggedStart(*[p.animate.move_to(medical_axes.c2p(x+1, 8*x + 5)) 
                         for p, x in zip(pills, np.linspace(1, 9, 5))],
            lag_ratio=0.3),
            run_time=2
        )
        self.wait(2)
        
        # Clear linear examples
        self.play(
            FadeOut(Group(linear_title, app1, sales_axes, sales_line, sales_dots, dollar_signs,
                          app2, medical_axes, medical_line, medical_dots, pills))
        )
        
        # ===== POLYNOMIAL REGRESSION APPLICATIONS =====
        poly_title = Text("Polynomial Regression", font_size=32, color=ORANGE_D)
        poly_title.move_to(UP*2.5)
        self.play(Write(poly_title))
        
        # Application 1: Economic Growth
        app3 = VGroup(
            Text("1. Economic Modeling", font_size=28, color=WHITE),
            Text("GDP growth over time\noften follows curves", font_size=22, color=GRAY_A)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(poly_title, DOWN, buff=1)
        
        # Create GDP growth curve
        # In the GDP growth section, modify these parts:

        # 1. First adjust the y-range of the axes to better fit your data
        gdp_axes = Axes(
            x_range=[0, 20, 5], 
            y_range=[0, 120, 30],  # Changed from [0,100,50] to [0,120,30]
            x_length=4, 
            y_length=2.5,
            axis_config={"color": WHITE}
        ).next_to(app3, DOWN, buff=0.5)

        # 2. Modify the GDP function to better fit within range
        def bounded_gdp(x):
            # Original coefficients were causing too steep growth
            raw = 0.1*x**3 - 1.5*x**2 + 6*x + 50  # Reduced coefficients
            return min(raw, 120)  # Matches our y-axis max

        # 3. Update the dots to match the new curve
        gdp_dots = VGroup(*[
            Dot(
                gdp_axes.c2p(
                    x, 
                    min(0.1*x**3 - 1.5*x**2 + 6*x + 50 + random.uniform(-5,5), 120)
                ), 
                color=ORANGE_D
            )
            for x in np.linspace(1, 19, 7)
        ])

        # 4. Create the curve with the bounded function
        gdp_curve = gdp_axes.plot(bounded_gdp, color=ORANGE_D)
        gdp_dots = VGroup(*[
                    Dot(gdp_axes.c2p(x, 0.2*x**3 - 2*x**2 + 8*x + 0 + random.uniform(-10,10)), 
                        color=ORANGE_D)
                    for x in np.linspace(1, 19, 7)
                ])
        
        self.play(FadeIn(app3))
        self.play(
            Create(gdp_axes),
            LaggedStart(*[GrowFromCenter(dot) for dot in gdp_dots], lag_ratio=0.2),
            Create(gdp_curve),
            run_time=2
        )
        self.wait(2)
        
        # Application 2: Engineering
        app4 = VGroup(
            Text("2. Engineering", font_size=28, color=WHITE),
            Text("Material stress-strain\nrelationships", font_size=22, color=GRAY_A)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(gdp_axes, DOWN, buff=1)
        
        # Create stress-strain curve
        eng_axes = Axes(
            x_range=[0, 10], y_range=[0, 120, 40],
            x_length=4, y_length=2.5,
            axis_config={"color": WHITE}
        ).next_to(app4, DOWN, buff=0.5)
        
        eng_curve = eng_axes.plot(
            lambda x: -0.5*x**3 + 6*x**2 + 10,
            color=ORANGE_D
        )
        eng_dots = VGroup(*[
            Dot(eng_axes.c2p(x, -0.5*x**3 + 6*x**2 + 10 + random.uniform(-8,8)), 
                color=ORANGE_D)
            for x in np.linspace(1, 9, 7)
        ])
        
        # Animate with breaking point
        material = Line(
            eng_axes.c2p(0, -15), eng_axes.c2p(10, -15),
            stroke_width=10, color=BLUE_D
        )
        crack = Line(
            eng_axes.c2p(5, -15), eng_axes.c2p(5.5, -15),
            stroke_width=3, color=RED_D
        ).set_opacity(0)
        
        self.play(FadeIn(app4))
        self.play(
            Create(eng_axes),
            LaggedStart(*[GrowFromCenter(dot) for dot in eng_dots], lag_ratio=0.2),
            Create(eng_curve),
            Create(material),
            run_time=2
        )
        self.wait(1)
        self.play(
            crack.animate.set_opacity(1),
            material.animate.set_color(RED_D),
            run_time=1.5
        )
        self.wait(2)
        
        # ===== FINAL COMPARISON =====
        self.play(FadeOut(Group(*[mob for mob in self.mobjects if mob != title])))
        
        comparison = VGroup(
            Text("When to Choose Each Method:", font_size=32, color=GREEN_D),
            BulletedList(
                "Linear: Constant rate relationships",
                "Linear: Quick, interpretable results",
                "Polynomial: Changing rate patterns",
                "Polynomial: Complex, curved data",
                font_size=26,
                buff=0.5
            )
        ).arrange(DOWN, aligned_edge=LEFT)
        
        self.play(FadeIn(comparison))
        self.wait(3)
        
        # Real-world examples summary
        self.play(FadeOut(comparison))
        app_title = Text("Common Applications:", font_size=30, color=PURPLE_D).to_edge(UP*0.1, buff=1)
        app_list = BulletedList(
            "Linear: Sales, dosage, pricing",
            "Linear: Simple trend forecasting",
            "Polynomial: Economic growth curves",
            "Polynomial: Engineering stress tests",
            font_size=24,
            buff=0.4
        ).next_to(app_title, DOWN-1.2, buff=0.5)

        # Then group them together
        examples = VGroup(app_title, app_list)
        self.play(Write(examples))
        self.wait(4)
        
        # Outro
        outro = Text("Thanks for watching!", font_size=40, gradient=(BLUE_D, GREEN_D))
        self.play(
            FadeOut(Group(*[mob for mob in self.mobjects if mob != title])),
            ReplacementTransform(title, outro)
        )
        self.wait(2)