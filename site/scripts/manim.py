"""Hierarchical robot planning: a progressively expanded Manim explanation.

Render: python -m manim -qm site/scripts/manim.py RoboticsWorldModel
Written for Manim Community. No LaTeX or external assets required.
Validated with a low-resolution Manim render.

This is an architecture animation, not a robotics simulator. Curves and costs
are illustrative. Slide 04 follows the DINO ROS bridge at pre-v1.0.0.
The surrounding slides remain a conceptual hierarchy; their separate state
decoder / local MPPI stages are not a literal trace of that bridge.
"""

from manim import *
import numpy as np


config.background_color = "#0B1220"
config.frame_width = 16
config.frame_height = 9

BG = "#0B1220"
PANEL = "#121E31"
INK = "#EDF3FF"
MUTED = "#A4B5CF"
CYAN = "#56D7EA"
PURPLE = "#B59AFF"
AMBER = "#F4BF66"
GREEN = "#86E2B4"
CORAL = "#F3A69B"
HORIZON = 36  # pre-v1.0.0 DINO ROS bridge default; plan_horizon is configurable.
PACE = 1.0  # Increase to slow all explanatory holds.
FONT = "Sans"  # Pango's portable sans-serif family.


def label(text, size=25, color=INK, width=None):
    # Pango positions ordinary glyphs reliably, unlike Unicode script glyphs
    # whose availability and baseline vary between fonts.
    mob = MarkupText(text, font=FONT, font_size=size, color=color, line_spacing=0.85)
    if width is not None and mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def card(title, subtitle="", width=3.2, height=1.15, color=CYAN):
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.16,
        stroke_color=color, stroke_width=2,
        fill_color=PANEL, fill_opacity=1,
    )
    title_mob = label(title, 25, color, width - 0.3)
    text = VGroup(title_mob)
    if subtitle:
        text.add(label(subtitle, 17, MUTED, width - 0.35))
    text.arrange(DOWN, buff=0.14).move_to(box)
    if text.height > height - 0.24:
        text.scale_to_fit_height(height - 0.24)
    return VGroup(box, text)


def arrow(a, b, color=MUTED, direction=RIGHT):
    if np.allclose(direction, DOWN):
        start, end = a.get_bottom(), b.get_top()
    else:
        start, end = a.get_right(), b.get_left()
    gap = np.linalg.norm(end - start)
    return Arrow(start, end, buff=min(0.12, gap * 0.12), color=color,
                 stroke_width=2.5, tip_length=0.13,
                 max_tip_length_to_length_ratio=0.35)


def elbow(points, color=MUTED):
    """Right-angle connection ending in an arrow, with no crossing through cards."""
    lines = VGroup(*[Line(a, b, color=color, stroke_width=2)
                     for a, b in zip(points[:-2], points[1:-1])])
    lines.add(Arrow(points[-2], points[-1], buff=0, color=color,
                    stroke_width=2, max_tip_length_to_length_ratio=0.15))
    return lines


class RoboticsWorldModel(Scene):
    def hold(self, seconds=2):
        self.wait(seconds * PACE)

    def heading(self, number, title, caption):
        new_header = VGroup(
            label(f"{number} / 07", 17, CYAN).move_to([-6.7, 4.0, 0]),
            label(title, 32, INK, 12.2).move_to([0, 3.6, 0]),
        )
        new_footer = label(caption, 19, MUTED, 14.6).move_to([0, -3.95, 0])
        if hasattr(self, "header"):
            self.play(ReplacementTransform(self.header, new_header),
                      ReplacementTransform(self.footer, new_footer), run_time=0.6)
        else:
            self.play(FadeIn(new_header), FadeIn(new_footer))
        self.header, self.footer = new_header, new_footer

    def pulse(self, connection, color=CYAN, duration=0.7):
        dot = Dot(connection.get_start(), radius=0.065, color=color)
        self.add(dot)
        self.play(MoveAlongPath(dot, connection), run_time=duration)
        self.remove(dot)

    def expand_stage(self, selected, title, color):
        """The existing stage box grows; its internals are revealed afterward."""
        self.play(FadeOut(self.stage_links),
                  *[FadeOut(s) for s in self.stages if s is not selected],
                  FadeOut(selected[1]), run_time=0.5)
        self.stage_saved = selected.copy()
        # selected's text has faded, so restore its opacity on the saved copy.
        self.stage_saved[1].set_opacity(1)
        big = RoundedRectangle(width=13.5, height=5.35, corner_radius=0.2,
                               color=color, fill_color=PANEL, fill_opacity=1)
        big.move_to([0, -0.05, 0])
        self.play(Transform(selected[0], big), run_time=1.2)
        self.detail_title = label(title, 22, color).move_to([0, 2.28, 0])
        self.play(FadeIn(self.detail_title))

    def collapse_stage(self, selected, details):
        self.play(FadeOut(details), FadeOut(self.detail_title), run_time=0.7)
        self.play(Transform(selected[0], self.stage_saved[0]), run_time=1.0)
        selected[1].set_opacity(1)
        self.play(FadeIn(selected[1]),
                  *[FadeIn(s) for s in self.stages if s is not selected],
                  FadeIn(self.stage_links), run_time=0.7)

    def construct(self):
        self.hierarchy()
        self.world_model_overview()
        self.encoders()
        self.latent_mpc()
        self.decoder()
        self.local_control()
        self.deployment_summary()

    def hierarchy(self):
        self.next_section("01 Hierarchical planning")
        self.heading("01", "One navigation task. Three planning scales.",
                     "GPS goal → mid-range subgoals → local controls")
        self.global_planner = card("LONG RANGE", "GPS goal / route waypoints",
                                   width=4.3, height=0.95, color=AMBER).move_to([0, 2.2, 0])
        self.mid_shell = RoundedRectangle(width=12.7, height=2.0,
            corner_radius=0.2, color=PURPLE, fill_color=PANEL, fill_opacity=0.4).move_to([0, 0.0, 0])
        self.mid_title = label("MID RANGE  /  choose one alternative", 20, PURPLE).move_to([0, 0.66, 0])
        self.astar = card("A*", "Graph search + path optimization", 3.65, 0.98, MUTED).move_to([-4.15, -0.2, 0])
        self.world = card("World-model planner", "Plan through imagined futures", 3.65, 0.98, PURPLE).move_to([0, -0.2, 0])
        self.neural = card("Neural policy", "Direct prediction; no explicit search", 3.65, 0.98, MUTED).move_to([4.15, -0.2, 0])
        self.local = card("SHORT RANGE / MPPI", "Sample controls • simulate • cost-weight", 4.3, 1.0, GREEN).move_to([0, -2.1, 0])
        self.top_link = arrow(self.global_planner, self.mid_shell, AMBER, DOWN)
        self.bottom_link = arrow(self.mid_shell, self.local, GREEN, DOWN)
        self.play(FadeIn(self.global_planner, shift=DOWN * 0.2))
        self.play(Create(self.top_link), Create(self.mid_shell), FadeIn(self.mid_title))
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.15) for x in
                               [self.astar, self.world, self.neural]], lag_ratio=0.25))
        self.play(Create(self.bottom_link), FadeIn(self.local))
        self.hold(3)
        self.play(self.astar.animate.set_opacity(0.25),
                  self.neural.animate.set_opacity(0.25),
                  Indicate(self.world, color=PURPLE, scale_factor=1.05))
        self.hold(1.5)

    def world_model_overview(self):
        self.next_section("02 Open the world model")
        self.heading("02", "Open the world-model planner",
                     "Encode the present. Predict possible futures. Decode useful subgoals.")
        self.play(*[FadeOut(x) for x in [self.global_planner, self.mid_shell,
            self.mid_title, self.astar, self.neural, self.local,
            self.top_link, self.bottom_link]], FadeOut(self.world[1]), run_time=0.8)
        target = RoundedRectangle(width=14.5, height=6.0, corner_radius=0.25,
            color=PURPLE, fill_color=PANEL, fill_opacity=0.4).move_to([0, -0.15, 0])
        self.play(Transform(self.world[0], target), run_time=1.5)
        self.world_label = label("World Model  /  Mid range Planner", 18, PURPLE).move_to([0, 2.55, 0])
        self.encoding = card("Observation encoding", "Vision + proprioception", 3.65, 1.55, CYAN).move_to([-4.6, 0, 0])
        self.mpc = card("Latent dynamics + MPC", "Configurable predictive horizon", 3.65, 1.55, PURPLE).move_to([0, 0, 0])
        self.decoding = card("State decoding", "Latent states → physical subgoals", 3.65, 1.55, AMBER).move_to([4.6, 0, 0])
        self.stages = [self.encoding, self.mpc, self.decoding]
        self.stage_links = VGroup(arrow(self.encoding, self.mpc, CYAN), arrow(self.mpc, self.decoding, PURPLE))
        self.play(FadeIn(self.world_label), LaggedStart(*[FadeIn(x) for x in self.stages], lag_ratio=0.2))
        self.play(Create(self.stage_links))
        for link in self.stage_links:
            self.pulse(link)
        self.hold(2.5)

    def encoders(self):
        self.next_section("03 Expand the encoders")
        self.heading("03", "Build the latent observation",
                     "Visual and proprioceptive features fuse into z<sub>t</sub>. Goal and action encodings condition planning.")
        self.play(FadeOut(self.world_label))
        self.expand_stage(self.encoding, "Observation Encoding  /  expanded", CYAN)
        visual = card("Camera observation", "Current image I<sub>t</sub>", 2.45, 0.85, CYAN).move_to([-5.15, 1.3, 0])
        proprio = card("Proprioception", "Angular velocity • linear\nacceleration • position", 2.45, 1.0, CYAN).move_to([-5.15, -0.05, 0])
        dino = card("DINOv2", "Visual encoder", 2.35, 0.85, CYAN).move_to([-2.15, 1.3, 0])
        prop_enc = card("Proprio encoder", "Sensor features", 2.35, 0.9, CYAN).move_to([-2.15, -0.05, 0])
        fuse = card("Fuse", "Joint representation", 2.05, 0.9, PURPLE).move_to([0.8, 0.65, 0])
        latent = card("Latent observation z<sub>t</sub>", "Input to predictive dynamics", 3.1, 1.0, PURPLE).move_to([4.2, 0.65, 0])
        goal = card("Subgoal position", "Target g<sub>t</sub>", 2.45, 0.75, CORAL).move_to([-5.15, -1.45, 0])
        goal_enc = card("Subgoal encoder", "Target latent z<sup>g</sup>", 2.35, 0.75, CORAL).move_to([-2.15, -1.45, 0])
        action = card("Candidate action a<sub>k</sub>", "Proposed during MPC", 2.5, 0.75, AMBER).move_to([0.85, -1.45, 0])
        action_enc = card("Action encoder", "Action latent e(a<sub>k</sub>)", 3.1, 0.75, AMBER).move_to([4.2, -1.45, 0])
        links = VGroup(arrow(visual, dino, CYAN), arrow(proprio, prop_enc, CYAN),
                       arrow(dino, fuse, CYAN), arrow(prop_enc, fuse, CYAN),
                       arrow(fuse, latent, PURPLE), arrow(goal, goal_enc, CORAL),
                       arrow(action, action_enc, AMBER))
        note = label("z<sup>g</sup> conditions the goal objective; e(a<sub>k</sub>) drives each dynamics transition.", 18, MUTED, 12).move_to([0, -2.28, 0])
        self.play(FadeIn(visual), FadeIn(proprio))
        self.play(FadeIn(dino), FadeIn(prop_enc), Create(VGroup(*links[:2])))
        self.play(Create(VGroup(*links[2:4])), FadeIn(fuse))
        self.play(Create(links[4]), FadeIn(latent))
        self.play(FadeIn(goal), FadeIn(goal_enc), Create(links[5]))
        self.play(FadeIn(action), FadeIn(action_enc), Create(links[6]), FadeIn(note))
        self.hold(4)
        details = VGroup(visual, proprio, dino, prop_enc, fuse, latent,
                         goal, goal_enc, action, action_enc, links, note)
        self.collapse_stage(self.encoding, details)
        self.hold(1)

    def latent_mpc(self):
        # Source: ../wm_planner at pre-v1.0.0 (ed831b9),
        # ros2/wm_planner_ros/python/wm_planner_ros/
        # dino_wm_planner_bridge_node.py: _cem_plan and _run_plan_and_publish.
        # WM.rollout uses observation/action history, with no goal input.
        self.next_section("04 Expand latent MPC")
        self.heading("04", "Search velocity sequences with CEM",
                     "DINO ROS bridge: goal enters the terminal XY cost; the winning velocities produce the published path.")
        self.expand_stage(self.mpc, "DINO world model + Cross-Entropy Method (CEM)", PURPLE)

        sample = card("1  Sample velocities", "N bounded sequences, each H × 3\nNormalized linear velocity", 3.65, 1.0, AMBER).move_to([-4.45, 1.25, 0])
        actions = card("2  Build action sequences", "Integrate position; keep orientation\nSet normalized angular velocity to 0", 3.65, 1.0, AMBER).move_to([0, 1.25, 0])
        rollout = card("3  World-model rollout", "Observation + action history\nPredict with a sliding latent history", 3.65, 1.0, PURPLE).move_to([4.45, 1.25, 0])
        score = card("4  Decode + score", "decode_act → final physical XY\nCost = distance to goal XY", 3.65, 1.0, CORAL).move_to([4.45, -0.35, 0])
        elite = card("5  Refit distribution", "Lowest-cost elites → mean / std\nSmooth update; repeat CEM iterations", 3.65, 1.0, PURPLE).move_to([0, -0.35, 0])
        best = card("6  Keep best sequence", "Lowest cost across all iterations\nReturn H × 3 normalized velocities", 3.65, 1.0, GREEN).move_to([-4.45, -0.35, 0])
        links = VGroup(
            arrow(sample, actions, AMBER), arrow(actions, rollout, PURPLE),
            arrow(rollout, score, PURPLE, DOWN),
            Arrow(score.get_left(), elite.get_right(), buff=0.12, color=CORAL),
            Arrow(elite.get_left(), best.get_right(), buff=0.12, color=GREEN),
        )
        repeat = elbow([elite.get_top(), [0, 0.4, 0],
                        [-4.45, 0.4, 0], sample.get_bottom()], PURPLE)
        output = card("Publish path", "Denormalize best velocities → integrate positions → transform frame → publish",
                      12.1, 0.8, GREEN).move_to([0, -1.95, 0])
        output_link = arrow(best, output, GREEN, DOWN)
        note = label(f"Default H = {HORIZON} (configurable) • goal is used for scoring • decoded positions are used inside CEM",
                     15, MUTED, 12.5).move_to([0, -2.65, 0])
        for node, link in [(sample, None), (actions, links[0]),
                           (rollout, links[1]), (score, links[2]),
                           (elite, links[3])]:
            self.play(FadeIn(node), *([] if link is None else [Create(link)]))
        self.play(Create(repeat))
        self.hold(2)
        self.play(Indicate(sample, color=AMBER), Indicate(elite, color=PURPLE))
        self.play(Create(links[4]), FadeIn(best))
        self.play(Create(output_link), FadeIn(output), FadeIn(note))
        self.hold(4)
        details = VGroup(sample, actions, rollout, score, elite, best,
                         links, repeat, output, output_link, note)
        self.collapse_stage(self.mpc, details)
        self.hold(1)

    def decoder(self):
        self.next_section("05 Decode subgoals")
        self.heading("05", "From latent predictions to physical subgoals",
                     "The state decoder produces target states; MPPI converts them into executable controls.")
        self.expand_stage(self.decoding, "STATE DECODING  /  expanded", AMBER)
        zs = VGroup(*[card(f"Latent subgoal {i + 1}", "Predicted latent state", 2.7, 0.8, PURPLE)
                      for i in range(3)]).arrange(DOWN, buff=0.4).move_to([-4.7, -0.05, 0])
        decoder = card("State decoder D", "Latent → physical state", 3.0, 1.2, AMBER).move_to([-0.6, -0.05, 0])
        ss = VGroup(*[card(f"Subgoal {i + 1}", "Position / heading", 2.6, 0.8, GREEN)
                      for i in range(3)]).arrange(DOWN, buff=0.4).move_to([4.2, -0.05, 0])
        links = VGroup(*[arrow(z, decoder, PURPLE) for z in zs],
                       *[arrow(decoder, s, AMBER) for s in ss])
        annotation = label("Three displayed subgoals summarize the planned latent trajectory.", 18, MUTED).move_to([0, -2.25, 0])
        self.play(LaggedStart(*[FadeIn(z) for z in zs], lag_ratio=0.2))
        self.play(Create(VGroup(*links[:3])), FadeIn(decoder))
        self.play(Create(VGroup(*links[3:])), LaggedStart(*[FadeIn(s) for s in ss], lag_ratio=0.2))
        self.play(FadeIn(annotation))
        self.hold(3.5)
        self.collapse_stage(self.decoding, VGroup(zs, decoder, ss, links, annotation))
        self.hold(1)

    def local_control(self):
        self.next_section("06 Execute and replan")
        self.heading("06", "Track locally. Observe again. Replan.",
                     "World-model MPC supplies a planned path; a downstream controller tracks it.")
        self.play(*[FadeOut(x) for x in self.stages], FadeOut(self.stage_links),
                  FadeOut(self.world[0]), run_time=0.7)
        goal = card("GPS goal", "Long range", 2.3, 0.85, AMBER).move_to([-5.45, 2.15, 0])
        wm = card("World-model MPC", f"Mid range • default H = {HORIZON}", 3.1, 0.85, PURPLE).move_to([-1.8, 2.15, 0])
        sg = card("Planned path", "Integrated selected velocities", 3.0, 0.85, AMBER).move_to([2.3, 2.15, 0])
        a1, a2 = arrow(goal, wm, AMBER), arrow(wm, sg, PURPLE)
        self.play(FadeIn(goal), FadeIn(wm), FadeIn(sg), Create(a1), Create(a2))
        shell = RoundedRectangle(width=6.1, height=3.7, corner_radius=0.2,
            color=GREEN, fill_color=PANEL, fill_opacity=0.65).move_to([-3.7, -0.6, 0])
        title = label("DOWNSTREAM MPPI CONTROLLER", 19, GREEN, 5.6).move_to([-3.7, 0.83, 0])
        sample = card("Sample controls", "Perturb a nominal sequence", 2.55, 0.75, GREEN).move_to([-5.1, -0.05, 0])
        simulate = card("Simulate dynamics", "Predict local motion", 2.55, 0.75, GREEN).move_to([-2.25, -0.05, 0])
        weight = card("Cost-weight + update", "Combine sampled controls", 2.55, 0.75, GREEN).move_to([-2.25, -1.25, 0])
        execute = card("Execute first control", "Shift horizon; repeat", 2.55, 0.75, GREEN).move_to([-5.1, -1.25, 0])
        local_links = VGroup(arrow(sample, simulate, GREEN), arrow(simulate, weight, GREEN, DOWN),
                            Arrow(weight.get_left(), execute.get_right(), buff=0.035,
                                  tip_length=0.1, color=GREEN,
                                  max_tip_length_to_length_ratio=0.35))
        # Keep the incoming path in the gutter, clear of both panel borders.
        # The reference path enters the controller's cost evaluation.
        input_link = elbow([sg.get_bottom(), [2.3, 1.5, 0],
                            [0, 1.5, 0], [0, -1.25, 0], weight.get_right()], AMBER)
        self.play(Create(shell), FadeIn(title))
        self.play(FadeIn(sample))
        self.play(Indicate(sample, color=GREEN))
        self.play(Create(local_links[0]))
        self.play(FadeIn(simulate))
        self.play(Indicate(simulate, color=GREEN))
        self.play(Create(local_links[1]))
        self.play(FadeIn(weight))
        self.play(Create(input_link))
        self.play(Indicate(weight, color=GREEN))
        self.play(Create(local_links[2]))
        self.play(FadeIn(execute))
        self.play(Indicate(execute, color=GREEN))
        # A schematic ground plane and sampled trajectories, not simulation data.
        ground = RoundedRectangle(width=6.15, height=3.7, corner_radius=0.2,
            color="#30405C", fill_color=PANEL, fill_opacity=0.3).move_to([3.8, -0.6, 0])
        ground_title = label("LOCAL TRAJECTORY", 21, GREEN).move_to([3.8, 0.83, 0])
        start = np.array([1.25, -1.8, 0])
        end = np.array([6.25, -0.15, 0])
        candidates = VGroup()
        for bend in np.linspace(-1, 1, 9):
            candidates.add(CubicBezier(start, start + [1.5, 1.3 + bend, 0],
                end + [-1.6, -0.8 + bend, 0], end, color=MUTED,
                stroke_width=1.4, stroke_opacity=0.45))
        path = CubicBezier(start, start + [1.5, 1.3, 0], end + [-1.6, -0.8, 0], end,
                           color=GREEN, stroke_width=4)
        waypoints = VGroup(*[Dot(path.point_from_proportion(t), radius=0.07, color=AMBER)
                            for t in [0.35, 0.68, 1.0]])
        rover = VGroup(RoundedRectangle(width=0.32, height=0.2, corner_radius=0.05,
            fill_color=CYAN, fill_opacity=1, stroke_color=INK),
            Dot(radius=0.035, color=BG).shift(RIGHT * 0.08)).move_to(start)

        # Finish the controller sequence before revealing the trajectory panel.
        # Add each object before animating it; opacity animations can otherwise
        # introduce unseen objects early (and fill the open candidate curves).
        self.play(Create(ground), FadeIn(ground_title))
        self.play(Create(candidates))
        self.play(candidates.animate.set_stroke(opacity=0.2))
        self.play(Create(path))
        self.play(FadeIn(waypoints))
        self.play(FadeIn(rover))
        control_link = elbow([execute.get_bottom(), [-5.1, -2.12, 0],
                              [0.55, -2.12, 0], start], GREEN)
        control_label = label("controls", 15, GREEN).move_to([-0.15, -2.32, 0])
        self.play(Create(control_link), FadeIn(control_label))
        self.play(MoveAlongPath(rover, path), run_time=3)
        feedback = elbow([rover.get_bottom(), [6.25, -3.05, 0],
                          [-7.15, -3.05, 0], [-7.15, 3.0, 0], [-1.8, 3.0, 0], wm.get_top()], CYAN)
        feedback_text = label("New camera + proprioception → updated latent observation → replan", 18, CYAN).move_to([0, -3.38, 0])
        self.play(Create(feedback), FadeIn(feedback_text), run_time=1.8)
        self.play(Indicate(wm, color=CYAN))
        self.hold(0.5)

    def deployment_summary(self):
        self.next_section("07 Simulation to zero-shot deployment")
        # Clear the preceding diagram while retaining the shared heading/footer.
        self.play(*[FadeOut(mob) for mob in list(self.mobjects)
                    if mob is not self.header and mob is not self.footer])
        self.heading("07", "From simulation to zero-shot deployment",
                     "Train in simulation. Deploy without additional training.")
        boxes = VGroup(*[
            VGroup(
                Rectangle(width=4.1, height=2.0, stroke_color=color,
                          stroke_width=2, fill_color=PANEL, fill_opacity=1),
                label(title, 25, color, width=3.8).move_to(ORIGIN),
            )
            for title, color in [
                ("Train in\nsimulation", CYAN),
                ("Zero-shot deployment\nin control settings", PURPLE),
                ("Zero-shot deployment\nin dynamic settings", GREEN),
            ]
        ]).arrange(RIGHT, buff=0.55).move_to([0, 0, 0])
        self.play(LaggedStart(*[FadeIn(box, shift=UP * 0.2) for box in boxes],
                              lag_ratio=0.3))
        self.hold(5)
