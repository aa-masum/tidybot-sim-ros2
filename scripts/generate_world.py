#!/usr/bin/env python3
"""Generates worlds/house.sdf: a furnished two-room house with hidden obstacles.

Layout: house 8 m x 5 m, x in [-4, 4], y in [-2.5, 2.5], divider wall at x = 0
with a 1.0 m doorway (y in [-0.5, 0.5]).

Open furniture (robot vacuum can drive underneath):
  * coffee table  - 4 legs, underside at 0.36 m
  * desk          - 2 side panels, underside at 0.71 m
Floor-standing furniture (sofa, armchair, bed, ...) is solid.
Only functional parts have collisions; decoration is visual-only.
The lidar of the cleaning robot is at z = 0.105 m, so it only sees legs/panels
of the open furniture, never the table tops.
"""
import os
import sys

SHADOWS = "false"   # "true" looks nicer but is slow without a GPU

HEADER = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="house">
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.6 0.6 0.58 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.4 0.3 -1</direction>
      <cast_shadows>false</cast_shadows>
    </light>
    <include><uri>model://ground_plane</uri></include>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.78 0.86 0.95 1</background>
      <shadows>{SHADOWS}</shadows>
    </scene>
    <physics type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
    </physics>
    <gui>
      <camera name="user_camera">
        <pose>0 -6.5 7 0 0.9 1.5708</pose>
      </camera>
    </gui>
"""

FOOTER = """  </world>
</sdf>
"""

# ---------------------------------------------------------------- colours
WALL = "0.50 0.58 0.68 1"        # slate blue walls
WALL_N = "0.38 0.45 0.55 1"      # divider core / wall top
TRIM = "0.97 0.97 0.95 1"
WOOD = "0.55 0.35 0.2 1"
WOOD_D = "0.35 0.22 0.13 1"
WOOD_L = "0.72 0.52 0.32 1"
FABRIC = "0.25 0.35 0.55 1"
FABRIC_L = "0.35 0.47 0.68 1"
DARK = "0.12 0.12 0.14 1"
GREY = "0.35 0.35 0.38 1"
GREEN = "0.1 0.5 0.15 1"
LEAF = "0.2 0.62 0.25 1"
TERRA = "0.72 0.38 0.25 1"
RED = "0.75 0.15 0.15 1"
CREAM = "0.92 0.9 0.82 1"
WHITE = "0.96 0.96 0.96 1"
SKY = "0.62 0.80 0.95 1"
BLANKET = "0.3 0.45 0.65 1"
BOOKS = ["0.7 0.15 0.15 1", "0.15 0.35 0.6 1", "0.85 0.7 0.2 1",
         "0.2 0.5 0.3 1", "0.5 0.25 0.55 1", "0.9 0.9 0.85 1"]


# ---------------------------------------------------------------- helpers
def geom_box(sx, sy, sz):
    return f"<box><size>{sx} {sy} {sz}</size></box>"


def geom_cyl(r, h):
    return f"<cylinder><radius>{r}</radius><length>{h}</length></cylinder>"


def geom_sph(r):
    return f"<sphere><radius>{r}</radius></sphere>"


def part(g, x, y, z, color, col, vis, yaw):
    return dict(g=g, x=x, y=y, z=z, color=color, col=col, vis=vis, yaw=yaw)


def B(x, y, z, sx, sy, sz, color, col=False, vis=True, yaw=0.0):
    return part(geom_box(sx, sy, sz), x, y, z, color, col, vis, yaw)


def C(x, y, z, r, h, color, col=False, vis=True):
    return part(geom_cyl(r, h), x, y, z, color, col, vis, 0.0)


def S(x, y, z, r, color, col=False):
    return part(geom_sph(r), x, y, z, color, col, True, 0.0)


def material(color):
    return (f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse>"
            f"<specular>0.08 0.08 0.08 1</specular></material>")


def model(name, x, y, yaw, parts):
    items = []
    for i, p in enumerate(parts):
        pose = f"<pose>{p['x']} {p['y']} {p['z']} 0 0 {p['yaw']}</pose>"
        if p["col"]:
            items.append(f'<collision name="c{i}">{pose}<geometry>{p["g"]}</geometry></collision>')
        if p["vis"]:
            items.append(f'<visual name="v{i}">{pose}<geometry>{p["g"]}</geometry>'
                         f'{material(p["color"])}</visual>')
    body = "\n        ".join(items)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 {yaw}</pose>
      <link name="link">
        {body}
      </link>
    </model>"""


def wall(name, x, y, sx, sy, color):
    return model(name, 0, 0, 0, [B(x, y, 0.5, sx, sy, 1.0, color, col=True)])


def plant(name, x, y):
    return model(name, x, y, 0.0, [
        C(0, 0, 0.15, 0.17, 0.3, TERRA, col=True),
        C(0, 0, 0.302, 0.15, 0.01, "0.2 0.12 0.08 1"),
        S(0, 0, 0.55, 0.24, GREEN, col=True),
        S(0.12, 0.05, 0.72, 0.14, LEAF),
        S(-0.1, -0.06, 0.74, 0.12, GREEN),
    ])


def lamp_parts(x, y, z):
    """Small table lamp, visual only. z = table top height."""
    return [
        C(x, y, z + 0.015, 0.08, 0.03, DARK),
        C(x, y, z + 0.15, 0.015, 0.25, DARK),
        C(x, y, z + 0.30, 0.11, 0.16, "0.95 0.88 0.65 1"),
    ]


def open_table(name, x, y, sx, sy, top, inset, top_col, leg_col, extra=None, leg=0.05):
    """Table on 4 legs, completely open underneath (collision: top slab + legs)."""
    parts_ = [B(0, 0, top - 0.02, sx, sy, 0.04, top_col, col=True)]
    for dx in (-(sx / 2 - inset), sx / 2 - inset):
        for dy in (-(sy / 2 - inset), sy / 2 - inset):
            parts_.append(B(dx, dy, (top - 0.04) / 2, leg, leg, top - 0.04, leg_col, col=True))
    return model(name, x, y, 0.0, parts_ + (extra or []))


parts = []

# ---------------------------------------------------------------- walls
parts.append(wall("wall_north_a", -2.05, 2.55, 4.1, 0.1, WALL))
parts.append(wall("wall_north_b", 2.05, 2.55, 4.1, 0.1, WALL))
parts.append(wall("wall_south_a", -2.05, -2.55, 4.1, 0.1, WALL))
parts.append(wall("wall_south_b", 2.05, -2.55, 4.1, 0.1, WALL))
parts.append(wall("wall_west", -4.05, 0.0, 0.1, 5.2, WALL))
parts.append(wall("wall_east", 4.05, 0.0, 0.1, 5.2, WALL))

for tag, yc in (("n", 1.5), ("s", -1.5)):
    parts.append(model(f"divider_{tag}", 0, 0, 0, [
        B(0.0, yc, 0.5, 0.1, 2.0, 1.0, WALL_N, col=True),
        B(-0.055, yc, 0.5, 0.01, 2.0, 1.0, WALL),
        B(0.055, yc, 0.5, 0.01, 2.0, 1.0, WALL),
    ]))

# ---------------------------------------------------------------- floors (same wood in both rooms)
tones = ["0.55 0.37 0.22 1", "0.50 0.33 0.20 1", "0.58 0.40 0.25 1"]
for name, cx in (("floor_a", -2.0), ("floor_b", 2.0)):
    planks = [B(cx, 0.0, 0.001, 4.0, 5.0, 0.002, "0.2 0.14 0.1 1")]
    for i in range(20):
        planks.append(B(cx, -2.375 + 0.25 * i, 0.0025, 4.0, 0.25, 0.005, tones[i % 3]))
    parts.append(model(name, 0, 0, 0, planks))

# Rugs (visual only)
parts.append(model("rug_a", -2.2, 0.7, 0, [
    C(0, 0, 0.008, 0.9, 0.012, "0.15 0.4 0.45 1"),
    C(0, 0, 0.009, 0.74, 0.014, "0.85 0.8 0.65 1"),
    C(0, 0, 0.010, 0.40, 0.016, "0.15 0.4 0.45 1"),
]))
parts.append(model("rug_b", 2.8, 0.1, 0, [
    B(0, 0, 0.008, 2.0, 1.0, 0.012, "0.45 0.3 0.5 1"),
    B(0, 0, 0.009, 1.8, 0.8, 0.014, "0.8 0.72 0.85 1"),
]))

# ---------------------------------------------------------------- trim & decor
trim = [
    B(0, 2.488, 0.04, 8.0, 0.024, 0.08, TRIM),
    B(0, -2.488, 0.04, 8.0, 0.024, 0.08, TRIM),
    B(3.988, 0, 0.04, 0.024, 5.0, 0.08, TRIM),
    B(-3.988, 0, 0.04, 0.024, 5.0, 0.08, TRIM),
    B(0, 0.53, 0.505, 0.16, 0.06, 1.01, TRIM),
    B(0, -0.53, 0.505, 0.16, 0.06, 1.01, TRIM),
]
for sx in (-0.062, 0.062):
    for yc in (1.5, -1.5):
        trim.append(B(sx, yc, 0.04, 0.024, 2.0, 0.08, TRIM))
parts.append(model("trim", 0, 0, 0, trim))


def window_ns(x, z):
    return [
        B(x, 2.495, z, 0.9, 0.02, 0.5, WHITE),
        B(x, 2.485, z, 0.8, 0.01, 0.4, SKY),
        B(x, 2.48, z, 0.03, 0.012, 0.4, WHITE),
        B(x, 2.48, z, 0.8, 0.012, 0.03, WHITE),
    ]


def window_ew(x_face, y, z):
    inward = -1.0 if x_face > 0 else 1.0
    return [
        B(x_face, y, z, 0.02, 0.9, 0.5, WHITE),
        B(x_face + inward * 0.01, y, z, 0.01, 0.8, 0.4, SKY),
        B(x_face + inward * 0.015, y, z, 0.012, 0.03, 0.4, WHITE),
        B(x_face + inward * 0.015, y, z, 0.012, 0.8, 0.03, WHITE),
    ]


decor = []
decor += window_ns(-1.0, 0.62)
decor += window_ns(2.6, 0.75)
decor += window_ew(3.995, 0.0, 0.62)
decor += window_ew(-3.995, -1.5, 0.62)
decor += [B(-2.2, 2.49, 0.8, 0.7, 0.02, 0.3, WOOD_D),
          B(-2.2, 2.48, 0.8, 0.62, 0.01, 0.22, "0.9 0.6 0.3 1"),
          B(-2.35, 2.475, 0.76, 0.25, 0.01, 0.12, "0.3 0.55 0.6 1"),
          B(-2.05, 2.475, 0.78, 0.2, 0.01, 0.15, "0.85 0.35 0.25 1")]
parts.append(model("decor", 0, 0, 0, decor))

# ---------------------------------------------------------------- Room A: living room
# Sofa along the north wall (solid). Free corridor 0.5 m in front of it,
# 0.88 m wide corner between sofa and west wall.
parts.append(model("sofa", -2.2, 2.0, 0.0, [
    B(0, 0, 0.15, 1.8, 0.8, 0.3, FABRIC, col=True),
    B(-0.43, -0.06, 0.35, 0.82, 0.6, 0.1, FABRIC_L, col=True),
    B(0.43, -0.06, 0.35, 0.82, 0.6, 0.1, FABRIC_L, col=True),
    B(0, 0.33, 0.475, 1.7, 0.14, 0.35, FABRIC, col=True),
    B(-0.43, 0.22, 0.5, 0.8, 0.1, 0.3, FABRIC_L),
    B(0.43, 0.22, 0.5, 0.8, 0.1, 0.3, FABRIC_L),
    B(-0.85, 0, 0.27, 0.14, 0.8, 0.54, FABRIC, col=True),
    B(0.85, 0, 0.27, 0.14, 0.8, 0.54, FABRIC, col=True),
    B(-0.62, -0.02, 0.5, 0.35, 0.1, 0.3, "0.85 0.6 0.25 1", yaw=0.3),
]))

parts.append(model("armchair", -3.45, 0.45, 0.0, [
    B(0, 0, 0.15, 0.7, 0.7, 0.3, FABRIC, col=True),
    B(0.04, 0, 0.35, 0.55, 0.55, 0.1, FABRIC_L, col=True),
    B(-0.28, 0, 0.45, 0.14, 0.7, 0.5, FABRIC, col=True),
    B(0, 0.31, 0.3, 0.7, 0.1, 0.4, FABRIC, col=True),
    B(0, -0.31, 0.3, 0.7, 0.1, 0.4, FABRIC, col=True),
]))

# Coffee table: 1.1 x 0.76 m, top at 0.40 m, 4 legs -> completely open underneath
parts.append(open_table("coffee_table", -2.2, 0.7, 1.1, 0.76, 0.40, 0.06, WOOD_L, WOOD_D, extra=[
    B(0.2, 0.0, 0.415, 0.25, 0.18, 0.03, "0.7 0.2 0.2 1", yaw=0.2),
    C(-0.25, 0.0, 0.46, 0.06, 0.12, "0.9 0.9 0.85 1"),
]))

parts.append(model("tv_unit", -3.8, -1.0, 0.0, [
    B(0, 0, 0.2, 0.3, 1.2, 0.4, DARK, col=True),
    B(0, 0, 0.72, 0.05, 1.0, 0.55, "0.03 0.03 0.04 1", col=True),
    B(0.03, 0, 0.72, 0.005, 0.95, 0.5, "0.1 0.15 0.3 1"),
    B(0, 0, 0.44, 0.1, 0.3, 0.03, GREY),
]))

books = []
for zs, n in ((0.05, 6), (0.37, 5), (0.69, 6)):
    for k in range(n):
        h = 0.18 + 0.03 * (k % 3)
        books.append(B(-0.38 + 0.12 * k, 0.0, zs + h / 2.0, 0.05, 0.22, h, BOOKS[(k + int(zs * 10)) % 6]))
parts.append(model("bookshelf", -1.5, -2.25, 0.0, [
    B(0, 0, 0.5, 1.0, 0.4, 1.0, WOOD, col=True, vis=False),
    B(-0.48, 0, 0.5, 0.04, 0.4, 1.0, WOOD),
    B(0.48, 0, 0.5, 0.04, 0.4, 1.0, WOOD),
    B(0, -0.18, 0.5, 1.0, 0.04, 1.0, WOOD_D),
    B(0, 0, 0.03, 0.96, 0.4, 0.04, WOOD),
    B(0, 0, 0.35, 0.96, 0.4, 0.04, WOOD),
    B(0, 0, 0.67, 0.96, 0.4, 0.04, WOOD),
    B(0, 0, 0.98, 1.0, 0.4, 0.04, WOOD),
] + books))

parts.append(plant("plant_a", -3.6, -2.1))

# Small side table (legs too close together for the robot)
parts.append(open_table("side_table", -0.5, 2.15, 0.4, 0.4, 0.49, 0.04, WOOD_L, WOOD_D,
                        extra=lamp_parts(0, 0, 0.49), leg=0.04))

# ---------------------------------------------------------------- Room B: bedroom
parts.append(model("bed", 2.8, 1.6, 0.0, [
    B(0, 0, 0.15, 2.0, 1.6, 0.3, WOOD_D, col=True),
    B(-0.02, 0, 0.375, 1.9, 1.5, 0.15, CREAM, col=True),
    B(0.96, 0, 0.55, 0.08, 1.6, 0.8, WOOD, col=True),
    B(0.72, -0.38, 0.49, 0.3, 0.55, 0.1, WHITE),
    B(0.72, 0.38, 0.49, 0.3, 0.55, 0.1, WHITE),
    B(-0.25, 0, 0.46, 1.3, 1.52, 0.04, BLANKET),
    B(-0.9, 0, 0.47, 0.25, 1.54, 0.05, "0.2 0.3 0.5 1"),
]))

parts.append(model("wardrobe", 3.7, -1.3, 0.0, [
    B(0, 0, 0.5, 0.5, 1.2, 1.0, WOOD, col=True),
    B(-0.255, -0.3, 0.5, 0.01, 0.56, 0.9, WOOD_L),
    B(-0.255, 0.3, 0.5, 0.01, 0.56, 0.9, WOOD_L),
    B(-0.262, -0.04, 0.5, 0.02, 0.03, 0.18, DARK),
    B(-0.262, 0.04, 0.5, 0.02, 0.03, 0.18, DARK),
]))

# Desk: 1.2 x 0.6 m, two side panels, open underneath (underside 0.71 m)
parts.append(model("desk", 1.2, 2.1, 0.0, [
    B(0, 0, 0.73, 1.2, 0.6, 0.04, WOOD_L, col=True),
    B(-0.58, 0, 0.355, 0.04, 0.6, 0.71, WOOD, col=True),
    B(0.58, 0, 0.355, 0.04, 0.6, 0.71, WOOD, col=True),
    B(-0.2, -0.05, 0.757, 0.3, 0.2, 0.015, GREY),
    B(-0.2, 0.07, 0.86, 0.3, 0.01, 0.2, DARK),
    B(0.4, 0.15, 0.78, 0.1, 0.1, 0.06, TERRA),
]))

# Chair pulled out from the desk (legs only on the floor)
chair = [
    B(0, 0, 0.43, 0.4, 0.4, 0.04, FABRIC, col=True),
    B(0, -0.18, 0.7, 0.4, 0.04, 0.5, DARK, col=True),
]
for dx in (-0.16, 0.16):
    for dy in (-0.16, 0.16):
        chair.append(B(dx, dy, 0.205, 0.04, 0.04, 0.41, GREY, col=True))
parts.append(model("chair", 1.2, 0.95, 0.0, chair))

parts.append(plant("plant_b", 0.5, -2.1))

dresser = [B(0, 0, 0.4, 1.0, 0.4, 0.8, WOOD, col=True)]
for dx in (-0.33, 0.0, 0.33):
    for dz in (0.58, 0.25):
        dresser.append(B(dx, 0.205, dz, 0.3, 0.01, 0.26, WOOD_L))
        dresser.append(B(dx, 0.215, dz, 0.08, 0.015, 0.02, DARK))
dresser.append(B(-0.3, 0, 0.82, 0.3, 0.2, 0.04, "0.8 0.75 0.6 1"))
parts.append(model("dresser", 2.2, -2.2, 0.0, dresser))

# ---------------------------------------------------------------- Hidden obstacles
parts.append(model("hidden_box", -0.9, 0.3, 0.0, [
    B(0, 0, 0.15, 0.3, 0.3, 0.3, RED, col=True),
    B(0, 0, 0.15, 0.04, 0.31, 0.31, "0.95 0.85 0.5 1"),
]))
parts.append(model("hidden_shoe", 1.0, -0.2, 0.5, [
    B(0, 0, 0.06, 0.25, 0.1, 0.12, RED, col=True, vis=False),
    B(0, 0, 0.02, 0.26, 0.1, 0.04, WHITE),
    B(-0.04, 0, 0.08, 0.14, 0.09, 0.08, RED),
    B(0.07, 0, 0.05, 0.12, 0.09, 0.06, "0.85 0.3 0.3 1"),
]))
parts.append(model("hidden_ball", 1.8, -1.0, 0.0, [
    S(0, 0, 0.1, 0.1, "0.15 0.4 0.85 1", col=True),
    B(0, 0, 0.1, 0.205, 0.03, 0.205, "0.95 0.8 0.2 1"),
]))

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "worlds", "house.sdf")
out_path = os.path.abspath(out_path)

with open(out_path, "w") as f:
    f.write(HEADER)
    f.write("\n".join(parts))
    f.write("\n")
    f.write(FOOTER)

print(f"World written to {out_path}")