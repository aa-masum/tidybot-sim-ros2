#!/usr/bin/env python3
"""Coverage planner + live coverage % for the two-room house (cleaning robot).

* Plans a lawn-mower style route on a PRIOR map (walls + furniture only).
* Open furniture (coffee table, desk) is modelled by its legs/panels only,
  so the robot plans routes underneath it.
* Drives the route with a simple pure-pursuit controller.
* Detects obstacles that are NOT on the prior map (hidden box/shoe/ball) from
  /scan and /scan_low, adds them to the planning grid and re-plans with A*.
* Publishes coverage %, a coverage map and the planned route.

/odom of Gazebo already reports WORLD coordinates (verified with
`gz model -m ugv -p`), so SPAWN_X/SPAWN_Y are 0.
"""
import collections
import datetime
import heapq
import json
import math
import os
import struct
import time
import zlib

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# ----------------------------------------------------------------- constants
SPAWN_X, SPAWN_Y = 0.0, 0.0           # odom already in world coordinates
USE_SLAM_POSE = False
RES = 0.05                            # grid resolution [m]
X0, Y0 = -4.0, -2.5                   # grid origin (world)
W, H = 160, 100                       # 8 m x 5 m
ROBOT_R = 0.19                        # robot radius 0.17 + 2 cm margin
CLEAN_R = 0.20                        # cleaning swath = 0.40 m
DYN_R = 0.20                          # inflation for newly found obstacles
HARD_STOP = 0.16                      # emergency stop distance (from centre)
SLOW_DIST = 0.50
NOSE = 0.16                           # low sensor offset from robot centre

# Positions of the hidden obstacles in the simulated world. NOT used by the
# robot or the planner - only the final report uses them to score how well the
# robot found the unmapped obstacles (simulation-only evaluation).
GROUND_TRUTH = [("box", -0.9, 0.3), ("shoe", 1.0, -0.2), ("ball", 1.8, -1.0)]
MATCH_TOL = 0.40                      # [m] max distance to count as "found"


def _legs(cx, cy, dx, dy, s=0.06):
    return [(cx + sx * dx, cy + sy * dy, s, s) for sx in (-1, 1) for sy in (-1, 1)]


# Footprints (cx, cy, size_x, size_y) - must match generate_world.py.
# Open furniture is represented by its legs / side panels only.
RECTS = (
    [(0.0, 1.5, 0.1, 2.0), (0.0, -1.5, 0.1, 2.0)]        # divider wall
    + [(-2.2, 2.0, 1.84, 0.8)]                           # sofa (solid)
    + [(-3.45, 0.45, 0.7, 0.7)]                          # armchair (solid)
    + _legs(-2.2, 0.7, 0.49, 0.32)                       # coffee table legs
    + [(-3.8, -1.0, 0.3, 1.2)]                           # tv unit
    + [(-1.5, -2.25, 1.0, 0.4)]                          # bookshelf
    + [(-3.6, -2.1, 0.48, 0.48)]                         # plant A
    + _legs(-0.5, 2.15, 0.16, 0.16)                      # side table legs
    + [(2.8, 1.6, 2.0, 1.6)]                             # bed (solid)
    + [(3.7, -1.3, 0.5, 1.2)]                            # wardrobe
    + [(0.62, 2.1, 0.05, 0.6), (1.78, 2.1, 0.05, 0.6)]   # desk side panels
    + _legs(1.2, 0.95, 0.16, 0.16)                       # chair legs
    + [(0.5, -2.1, 0.48, 0.48)]                          # plant B
    + [(2.2, -2.2, 1.0, 0.4)]                            # dresser
)

LANE_ROWS = np.linspace(5, 94, 14).round().astype(int)   # grid rows of the lanes


# ----------------------------------------------------------------- grid helpers
def to_cell(x, y):
    return int(math.floor((x - X0) / RES)), int(math.floor((y - Y0) / RES))


def to_world(ix, iy):
    return X0 + (ix + 0.5) * RES, Y0 + (iy + 0.5) * RES


def in_bounds(ix, iy):
    return 0 <= ix < W and 0 <= iy < H


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


_DISC = {}


def get_disc(r_m):
    key = round(r_m, 3)
    if key not in _DISC:
        n = int(math.ceil(r_m / RES))
        pts = [(dx, dy) for dy in range(-n, n + 1) for dx in range(-n, n + 1)
               if dx * dx + dy * dy <= (r_m / RES) ** 2]
        _DISC[key] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return _DISC[key]


def dilate(grid, r_m):
    out = grid.copy()
    dxs, dys = get_disc(r_m)
    for dx, dy in zip(dxs.tolist(), dys.tolist()):
        if dx == 0 and dy == 0:
            continue
        dst_y = slice(max(0, dy), min(H, H + dy))
        src_y = slice(max(0, -dy), min(H, H - dy))
        dst_x = slice(max(0, dx), min(W, W + dx))
        src_x = slice(max(0, -dx), min(W, W - dx))
        out[dst_y, dst_x] |= grid[src_y, src_x]
    return out


def stamp(grid, ix, iy, r_m):
    dxs, dys = get_disc(r_m)
    xs, ys = ix + dxs, iy + dys
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    grid[ys[ok], xs[ok]] = True


def build_prior():
    """Occupancy grid of walls + furniture (no hidden obstacles)."""
    xs = X0 + (np.arange(W) + 0.5) * RES
    ys = Y0 + (np.arange(H) + 0.5) * RES
    xx, yy = np.meshgrid(xs, ys)
    g = (xx < -3.95) | (xx > 3.95) | (yy < -2.45) | (yy > 2.45)
    for cx, cy, sx, sy in RECTS:
        g |= (np.abs(xx - cx) <= sx / 2) & (np.abs(yy - cy) <= sy / 2)
    return g


def reachable(blocked, start):
    """4-connected flood fill of free cells from start."""
    seen = np.zeros((H, W), bool)
    bl = blocked.tolist()
    dq = collections.deque([start])
    seen[start[1], start[0]] = True
    while dq:
        x, y = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not bl[ny][nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                dq.append((nx, ny))
    return seen


def line_free(blocked, a, b, skip=0.0):
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (RES * 0.5)))
    for i in range(n + 1):
        t = i / n
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if skip and math.hypot(x - a[0], y - a[1]) < skip:
            continue
        ix, iy = to_cell(x, y)
        if not in_bounds(ix, iy) or blocked[iy, ix]:
            return False
    return True


def nearest_free(blocked, c, max_r=30):
    cx, cy = c
    best, bd = None, 1e9
    for r in range(max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny) and not blocked[ny, nx]:
                    d = dx * dx + dy * dy
                    if d < bd:
                        bd, best = d, (nx, ny)
        if best is not None:
            return best
    return None


def astar(blocked, start, goal):
    """8-connected A* on cell coordinates (ix, iy). Returns list of cells."""
    if start == goal:
        return [start]
    bl = blocked.tolist()
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return (dx + dy) - 0.586 * min(dx, dy)

    openq = [(h(start), 0.0, start)]
    came = {}
    gs = {start: 0.0}
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > gs.get(cur, 1e18):
            continue
        for dx, dy, c in moves:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < W and 0 <= ny < H) or bl[ny][nx]:
                continue
            if dx and dy and (bl[cur[1]][nx] or bl[ny][cur[0]]):
                continue                       # no corner cutting
            ng = g + c
            if ng < gs.get((nx, ny), 1e18):
                gs[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(openq, (ng + h((nx, ny)), ng, (nx, ny)))
    return None


def smooth(blocked, pts):
    """Greedy line-of-sight shortcut. pts[0] is the start (not returned)."""
    out, i = [], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not line_free(blocked, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def make_segments(blocked, x_lo, x_hi):
    """Free runs along each lane (rows LANE_ROWS) inside [x_lo, x_hi]."""
    segs = []
    for iy in LANE_ROWS.tolist():
        run = None
        for ix in range(W + 1):
            ok = False
            if ix < W:
                x, _ = to_world(ix, iy)
                ok = x_lo <= x <= x_hi and not blocked[iy, ix]
            if ok:
                run = [ix, ix] if run is None else [run[0], ix]
            elif run is not None:
                if (run[1] - run[0]) * RES >= 0.2:
                    segs.append((to_world(run[0], iy), to_world(run[1], iy)))
                run = None
    return segs


def order_segments(segs, pos):
    """Nearest-neighbour ordering; returns waypoint list and end position."""
    segs, out = list(segs), []
    while segs:
        best, bd = None, 1e9
        for i, (a, b) in enumerate(segs):
            for p, q in ((a, b), (b, a)):
                d = math.hypot(p[0] - pos[0], p[1] - pos[1])
                if d < bd:
                    bd, best = d, (i, p, q)
        i, p, q = best
        segs.pop(i)
        out += [p, q]
        pos = q
    return out, pos


def sector_min(scan, lo_deg, hi_deg):
    ranges = np.array(scan.ranges, dtype=float)
    ranges[~np.isfinite(ranges)] = scan.range_max
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    mask = (angles >= math.radians(lo_deg)) & (angles <= math.radians(hi_deg))
    return float(ranges[mask].min()) if mask.any() else scan.range_max


# ----------------------------------------------------------------- obstacle log & report
def label_clusters(mask):
    """8-connected components of a boolean grid -> list of [(ix, iy), ...]."""
    seen = np.zeros_like(mask)
    comps = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        comp, dq = [], collections.deque([(x, y)])
        seen[y, x] = True
        while dq:
            cx, cy = dq.popleft()
            comp.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        dq.append((nx, ny))
        comps.append(comp)
    return comps


def find_obstacles(dyn_raw, first_time):
    """Group the detected hit cells into obstacles (centre, size, first seen)."""
    obstacles = []
    for comp in label_clusters(dilate(dyn_raw, 0.10)):
        cells = [(x, y) for x, y in comp if dyn_raw[y, x]]
        if len(cells) < 4:
            continue
        pts = [to_world(x, y) for x, y in cells]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ft = float(np.nanmin([first_time[y, x] for x, y in cells]))
        obstacles.append(dict(x=sum(xs) / len(xs), y=sum(ys) / len(ys),
                              width=max(xs) - min(xs) + RES,
                              depth=max(ys) - min(ys) + RES,
                              cells=len(cells), first_seen=ft))
    obstacles.sort(key=lambda o: o['first_seen'])
    for i, o in enumerate(obstacles, 1):
        o['id'] = i
    return obstacles


def match_truth(obstacles, truth=GROUND_TRUTH, tol=MATCH_TOL):
    """Score the detections against the (simulation-only) ground truth."""
    results, used = [], set()
    for name, gx, gy in truth:
        best, bd = None, 1e9
        for o in obstacles:
            d = math.hypot(o['x'] - gx, o['y'] - gy)
            if d < bd and o['id'] not in used:
                best, bd = o, d
        found = best is not None and bd <= tol
        if found:
            used.add(best['id'])
        results.append(dict(name=name, x=gx, y=gy, found=found,
                            matched_id=best['id'] if found else None,
                            error_m=round(bd, 2) if found else None))
    false_pos = [o['id'] for o in obstacles if o['id'] not in used]
    return results, false_pos


def save_png(path, rgb):
    """Minimal PNG writer (fallback when matplotlib is not installed)."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)

    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)


def render_png(path, stats, obstacles, truth, raw, visited, coverable, trace):
    img = np.full((H, W, 3), 0.93)
    img[coverable & ~visited] = (1.0, 0.65, 0.25)
    img[coverable & visited] = (0.45, 0.75, 0.55)
    img[raw] = (0.18, 0.18, 0.22)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        big = np.kron((img * 255).astype(np.uint8), np.ones((6, 6, 1), np.uint8))[::-1].copy()
        for o in obstacles:      # red squares (no text without matplotlib)
            cx = int((o['x'] - X0) / RES * 6)
            cy = int((Y0 + H * RES - o['y']) / RES * 6)
            big[max(cy - 9, 0):cy + 9, max(cx - 9, 0):cx + 9] = (230, 30, 30)
        save_png(path, big)
        return 'fallback (matplotlib not installed)'

    fig = plt.figure(figsize=(13, 6.6))
    ax = fig.add_axes([0.04, 0.08, 0.60, 0.84])
    ax.imshow(img, origin='lower', extent=(X0, X0 + W * RES, Y0, Y0 + H * RES),
              interpolation='nearest')
    if trace:
        tx, ty = zip(*trace)
        ax.plot(tx, ty, color='#1f4e9c', lw=0.6, alpha=0.7)
    for o in obstacles:
        ax.scatter(o['x'], o['y'], s=150, marker='X', c='red', edgecolors='white',
                   linewidths=1.2, zorder=5)
        ax.annotate(f"#{o['id']}", (o['x'], o['y']), xytext=(7, 7),
                    textcoords='offset points', color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='red', ec='none'))
    ax.set_title(f"Cleaning run - coverage {stats['coverage_final']:.1f}%", fontsize=13)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(handles=[Patch(color=(0.45, 0.75, 0.55), label='cleaned'),
                       Patch(color=(1.0, 0.65, 0.25), label='missed'),
                       Patch(color=(0.18, 0.18, 0.22), label='furniture / wall'),
                       plt.Line2D([0], [0], color='#1f4e9c', lw=1, label='robot path'),
                       plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                                  markersize=11, label='unmapped obstacle')],
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False)

    lines = ['RUN SUMMARY', '-' * 34,
             f"Sim time        : {stats['sim_duration_s'] / 60:5.1f} min",
             f"Distance        : {stats['distance_m']:5.1f} m",
             f"Avg speed       : {stats['avg_speed_mps']:5.2f} m/s",
             f"Coverage final  : {stats['coverage_final']:5.1f} %",
             f"After lanes     : {stats['coverage_after_lanes']:5.1f} %",
             f"Cleanup goals   : {stats['cleanup_goals']}",
             f"Re-plans        : {stats['replans']}",
             f"Stuck recoveries: {stats['stuck_events']}", '',
             f"UNMAPPED OBSTACLES: {len(obstacles)}", '-' * 34]
    for o in obstacles:
        lines.append(f"#{o['id']} ({o['x']:5.2f},{o['y']:5.2f}) "
                     f"{o['width'] * 100:3.0f}x{o['depth'] * 100:3.0f}cm t={o['first_seen']:.0f}s")
    if truth:
        lines += ['', 'CHECK vs SIMULATION', '-' * 34]
        for t in truth:
            res = (f"found as #{t['matched_id']} (err {t['error_m']:.2f} m)"
                   if t['found'] else 'NOT FOUND')
            lines.append(f"{t['name']:5s} ({t['x']:5.2f},{t['y']:5.2f}) {res}")
    ax2 = fig.add_axes([0.67, 0.08, 0.31, 0.84])
    ax2.axis('off')
    ax2.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=9.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return 'ok'


def write_report(outdir, stats, obstacles, truth, false_pos, raw, visited, coverable, trace):
    os.makedirs(outdir, exist_ok=True)
    files = []

    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(dict(stats=stats, obstacles=obstacles, ground_truth_check=truth,
                       unmatched_detections=false_pos), f, indent=2)
    files.append('report.json')

    md = ['# Cleaning run report', '',
          f"Generated: {stats['generated']}", '', '## Summary', '',
          '| Metric | Value |', '|---|---|',
          f"| Final coverage | {stats['coverage_final']:.1f} % |",
          f"| Coverage after lane phase | {stats['coverage_after_lanes']:.1f} % |",
          f"| Simulated time | {stats['sim_duration_s'] / 60:.1f} min |",
          f"| Distance driven | {stats['distance_m']:.1f} m |",
          f"| Average speed | {stats['avg_speed_mps']:.2f} m/s |",
          f"| Cleanup goals | {stats['cleanup_goals']} |",
          f"| Re-plans (new obstacle) | {stats['replans']} |",
          f"| Stuck recoveries | {stats['stuck_events']} |", '',
          f"## Unmapped obstacles found: {len(obstacles)}", '']
    if obstacles:
        md += ['| # | x [m] | y [m] | size [cm] | first seen [s] |', '|---|---|---|---|---|']
        for o in obstacles:
            md.append(f"| {o['id']} | {o['x']:.2f} | {o['y']:.2f} | "
                      f"{o['width'] * 100:.0f} x {o['depth'] * 100:.0f} | {o['first_seen']:.0f} |")
    if truth:
        n_found = sum(1 for t in truth if t['found'])
        md += ['', f"## Check against simulation ground truth ({n_found}/{len(truth)} found)", '',
               '| Object | True x | True y | Result |', '|---|---|---|---|']
        for t in truth:
            res = f"found as #{t['matched_id']}, error {t['error_m']:.2f} m" if t['found'] else 'not found'
            md.append(f"| {t['name']} | {t['x']:.2f} | {t['y']:.2f} | {res} |")
        md.append('')
        md.append(f"Detections matching no known object (false alarms): {len(false_pos)}")
    md += ['', '![map](report.png)', '']
    with open(os.path.join(outdir, 'report.md'), 'w') as f:
        f.write('\n'.join(md))
    files.append('report.md')

    mode = render_png(os.path.join(outdir, 'report.png'), stats, obstacles, truth,
                      raw, visited, coverable, trace)
    files.append(f'report.png [{mode}]')
    return files


# ----------------------------------------------------------------- node
class CoveragePlanner(Node):
    def __init__(self):
        super().__init__('coverage_planner')
        self.declare_parameter('max_speed', 0.25)
        self.declare_parameter('turn_speed', 0.9)
        self.declare_parameter('report_dir', '~/ugv_reports')
        self.max_speed = self.get_parameter('max_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value

        # prior knowledge
        self.raw = build_prior()
        self.prior_infl = dilate(self.raw, ROBOT_R)
        self.explained = dilate(self.raw, 0.30)
        self.reach = ~self.prior_infl
        self.coverable = dilate(self.reach, CLEAN_R) & ~self.raw

        # unmapped obstacles found at run time
        self.dyn_count = np.zeros((H, W), np.int16)
        self.dyn_raw = np.zeros((H, W), bool)
        self.dyn_infl = np.zeros((H, W), bool)
        self.dyn_excl = np.zeros((H, W), bool)
        self.blocked = self.prior_infl.copy()
        self.dyn_changed = False
        self.last_replan = 0.0

        # coverage
        self.visited = np.zeros((H, W), bool)
        self.cdx, self.cdy = get_disc(CLEAN_R)

        # state
        self.pos = None
        self.yaw = 0.0
        self.wz = 0.0
        self.prog_pos = None
        self.prog_t = 0.0
        self.recover_until = 0.0
        self.tf_buf = Buffer()
        self.tf_lis = TransformListener(self.tf_buf, self)
        self.scan = None
        self.low = None
        self.targets = None
        self.t_idx = 0
        self.phase = 'lanes'          # lanes -> cleanup -> done
        self.leg_path = []
        self.leg_idx = 0
        self.cur_goal = None
        self.bad_goals = []
        self.cleanup_count = 0
        self.pct = 0.0
        self.last_log = 0.0

        # run statistics (for the final report)
        self.t_start = None
        self.wall_start = time.time()
        self.dist = 0.0
        self.last_pos = None
        self.trace = []
        self.last_trace_t = -1.0
        self.n_replans = 0
        self.n_stuck = 0
        self.pct_lanes = 0.0
        self.dyn_first = np.full((H, W), np.nan, np.float32)
        self.dyn_ver = 0
        self.markers_ver = -1
        self.report_done = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan_low', self.on_low, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_pct = self.create_publisher(Float32, '/coverage_percent', 10)
        self.pub_map = self.create_publisher(OccupancyGrid, '/coverage_map', latched)
        self.pub_path = self.create_publisher(Path, '/coverage_path', latched)
        self.pub_obst = self.create_publisher(MarkerArray, '/found_obstacles', latched)
        self.create_timer(0.1, self.control)
        self.create_timer(1.0, self.report)
        self.get_logger().info('Coverage planner started')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- callbacks
    def on_odom(self, msg):
        p = msg.pose.pose
        self.wz = msg.twist.twist.angular.z
        x, y = p.position.x, p.position.y
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        try:   # SLAM pose disabled: map frame may not equal the world frame
            if not USE_SLAM_POSE:
                raise RuntimeError('disabled')
            tf = self.tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time())
            x, y = tf.transform.translation.x, tf.transform.translation.y
            q = tf.transform.rotation
            yaw = 2.0 * math.atan2(q.z, q.w)
        except Exception:
            pass
        self.pos = (x + SPAWN_X, y + SPAWN_Y)
        self.yaw = yaw

    def on_scan(self, msg):
        self.scan = msg
        self.mark_hits(msg, 0.0)

    def on_low(self, msg):
        self.low = msg
        self.mark_hits(msg, NOSE)

    def mark_hits(self, scan, offset):
        """Hits that the prior map cannot explain become new obstacles."""
        if self.pos is None or abs(self.wz) > 0.15:
            return      # no mapping while rotating
        r = np.array(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > 0.12) & (r < min(scan.range_max - 0.05, 1.5))
        if not ok.any():
            return
        r, a = r[ok], a[ok]
        px = self.pos[0] + offset * math.cos(self.yaw)
        py = self.pos[1] + offset * math.sin(self.yaw)
        wx = px + r * np.cos(self.yaw + a)
        wy = py + r * np.sin(self.yaw + a)
        ix = np.floor((wx - X0) / RES).astype(int)
        iy = np.floor((wy - Y0) / RES).astype(int)
        inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix, iy = ix[inb], iy[inb]
        un = ~self.explained[iy, ix]
        if not un.any():
            return
        cells = np.unique(iy[un] * W + ix[un])
        self.dyn_count.reshape(-1)[cells] += 1
        new = (self.dyn_count >= 3) & ~self.dyn_raw
        if not new.any():
            return
        self.dyn_raw |= new
        self.dyn_first[new] = self.now()
        self.dyn_ver += 1
        for cy, cx in zip(*np.nonzero(new)):
            stamp(self.dyn_infl, int(cx), int(cy), DYN_R)
            stamp(self.dyn_excl, int(cx), int(cy), 0.12)
        self.blocked = self.prior_infl | self.dyn_infl
        self.dyn_changed = True
        x, y = to_world(int(cx), int(cy))
        self.get_logger().info(
            f'Unmapped obstacle near ({x:.2f}, {y:.2f}) -> will re-plan',
            throttle_duration_sec=2.0)

    # ------------------------------------------------------------- planning
    def build_plan(self):
        s = to_cell(*self.pos)
        if self.prior_infl[s[1], s[0]]:
            s = nearest_free(self.prior_infl, s)
        self.reach = reachable(self.prior_infl, s)
        self.coverable = dilate(self.reach, CLEAN_R) & ~self.raw
        blk = self.prior_infl | ~self.reach
        seg_a = make_segments(blk, -4.0, -0.15)
        seg_b = make_segments(blk, 0.15, 4.0)
        wa, end_a = order_segments(seg_a, self.pos)
        wb, _ = order_segments(seg_b, end_a)
        self.targets = wa + wb
        self.t_idx = 0
        self.get_logger().info(f'Planned {len(self.targets) // 2} lane segments '
                               f'({len(seg_a)} in room A, {len(seg_b)} in room B); '
                               f'{int(self.coverable.sum())} cells coverable')
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.targets:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = x - SPAWN_X
            ps.pose.position.y = y - SPAWN_Y
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

    def plan_leg(self, goal):
        s = to_cell(*self.pos)
        g = to_cell(*goal)
        if not in_bounds(*s) or not in_bounds(*g):
            return False
        s0 = s
        if self.blocked[s[1], s[0]]:
            s = nearest_free(self.blocked, s)
        if self.blocked[g[1], g[0]]:
            g = nearest_free(self.blocked, g)
        if s is None or g is None:
            return False
        cells = astar(self.blocked, s, g)
        if cells is None:
            return False
        pts = smooth(self.blocked, [to_world(*c) for c in cells])
        if s != s0:
            pts.insert(0, to_world(*s))
        if not pts:
            return False
        self.leg_path, self.leg_idx = pts, 0
        return True

    def replan(self):
        self.n_replans += 1
        self.get_logger().info('Re-planning around new obstacle')
        if not self.plan_leg(self.cur_goal):
            self.get_logger().warn('Re-plan failed, skipping this target')
            self.leg_path, self.leg_idx = [], 0

    def path_clear(self, upto=2.0):
        pts = [self.pos] + self.leg_path[self.leg_idx:]
        dist = 0.0
        for k, (a, b) in enumerate(zip(pts, pts[1:])):
            if not line_free(self.blocked, a, b, skip=0.3 if k == 0 else 0.0):
                return False
            dist += math.hypot(b[0] - a[0], b[1] - a[1])
            if dist > upto:
                break
        return True

    def cleanup_goal(self):
        """Nearest spot where enough un-cleaned floor is still within reach."""
        unc = self.coverable & ~self.visited & ~self.dyn_excl
        ii = np.zeros((H + 1, W + 1), np.int32)
        ii[1:, 1:] = unc.cumsum(0).cumsum(1)
        k = int(CLEAN_R / RES)
        best, bd = None, 1e9
        for iy in range(k, H - k, 3):
            for ix in range(k, W - k, 3):
                if self.blocked[iy, ix] or not self.reach[iy, ix]:
                    continue
                cnt = (ii[iy + k + 1, ix + k + 1] - ii[iy - k, ix + k + 1]
                       - ii[iy + k + 1, ix - k] + ii[iy - k, ix - k])
                if cnt < 10:
                    continue
                x, y = to_world(ix, iy)
                if any(math.hypot(x - bx, y - by) < 0.2 for bx, by in self.bad_goals):
                    continue
                d = math.hypot(x - self.pos[0], y - self.pos[1])
                if d < bd:
                    bd, best = d, (x, y)
        if best is not None:
            self.bad_goals.append(best)
        return best

    def advance(self):
        """Pick the next goal and plan a leg to it. False when finished."""
        while True:
            if self.phase == 'lanes':
                if self.t_idx < len(self.targets):
                    goal = self.targets[self.t_idx]
                    self.t_idx += 1
                else:
                    self.phase = 'cleanup'
                    self.pct_lanes = self.pct
                    self.get_logger().info(f'Lanes finished at {self.pct:.1f}% - cleanup pass')
                    continue
            elif self.phase == 'cleanup':
                goal = self.cleanup_goal() if self.cleanup_count < 80 else None
                if goal is None:
                    self.phase = 'done'
                    self.get_logger().info(f'DONE. Final coverage: {self.pct:.1f}%')
                    self.finish()
                    return False
                self.cleanup_count += 1
            else:
                return False
            self.cur_goal = goal
            if self.plan_leg(goal):
                return True

    # ------------------------------------------------------------- control
    def mark_visited(self):
        ix, iy = to_cell(*self.pos)
        xs, ys = ix + self.cdx, iy + self.cdy
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        self.visited[ys[ok], xs[ok]] = True

    def control(self):
        cmd = Twist()
        if self.pos is None or self.scan is None or self.low is None:
            self.pub.publish(cmd)
            return
        self.mark_visited()
        if self.phase == 'done':
            self.pub.publish(cmd)
            return
        if self.targets is None:
            self.build_plan()

        t = self.now()
        if self.t_start is None:
            self.t_start = t
        if self.last_pos is not None:
            self.dist += math.hypot(self.pos[0] - self.last_pos[0], self.pos[1] - self.last_pos[1])
        self.last_pos = self.pos
        if t - self.last_trace_t >= 0.5 and len(self.trace) < 20000:
            self.last_trace_t = t
            self.trace.append(self.pos)
        if t < self.recover_until:
            cmd.linear.x = -0.08
            self.pub.publish(cmd)
            return
        if self.prog_pos is None or math.hypot(self.pos[0] - self.prog_pos[0],
                                               self.pos[1] - self.prog_pos[1]) > 0.2:
            self.prog_pos, self.prog_t = self.pos, t
        elif t - self.prog_t > 8.0:
            self.n_stuck += 1
            self.get_logger().warn('Stuck - backing up and skipping this target')
            self.recover_until = t + 1.5
            self.leg_path, self.leg_idx = [], 0
            self.prog_pos = None
            self.pub.publish(cmd)
            return

        if self.dyn_changed and self.leg_path and t - self.last_replan > 1.0:
            self.dyn_changed = False
            if not self.path_clear():
                self.last_replan = t
                self.replan()

        if self.leg_idx >= len(self.leg_path):
            if not self.advance():
                self.pub.publish(cmd)
                return

        tx, ty = self.leg_path[self.leg_idx]
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        d = math.hypot(dx, dy)
        final = self.leg_idx == len(self.leg_path) - 1
        if d < (0.10 if final else 0.15):
            self.leg_idx += 1
            self.pub.publish(cmd)
            return

        err = wrap(math.atan2(dy, dx) - self.yaw)
        front = min(sector_min(self.scan, -25, 25),
                    sector_min(self.low, -35, 35) + NOSE)

        if front < HARD_STOP and abs(err) < 0.6:
            # emergency: something is right in front -> turn to the freer side
            left = min(sector_min(self.scan, 20, 90), sector_min(self.low, 15, 35) + NOSE)
            right = min(sector_min(self.scan, -90, -20), sector_min(self.low, -35, -15) + NOSE)
            cmd.angular.z = self.turn_speed if left >= right else -self.turn_speed
            self.dyn_changed = True
            self.pub.publish(cmd)
            return

        if abs(err) > 0.6:
            cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 1.5 * err))
        else:
            v = self.max_speed * (1.0 - abs(err) / 0.9)
            v = min(v, 0.6 * d + 0.05)
            if front < SLOW_DIST:
                v = min(v, max(0.06, self.max_speed * (front - HARD_STOP) / (SLOW_DIST - HARD_STOP)))
            cmd.linear.x = v
            cmd.angular.z = max(-self.turn_speed, min(self.turn_speed, 1.8 * err))
        self.pub.publish(cmd)

    # ------------------------------------------------------------- reporting
    def report(self):
        mask = self.coverable & ~self.dyn_excl
        den = int(mask.sum())
        num = int((self.visited & mask).sum())
        self.pct = 100.0 * num / max(den, 1)
        self.pub_pct.publish(Float32(data=float(self.pct)))

        g = np.full((H, W), -1, np.int8)
        g[self.visited & self.coverable] = 70
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = RES
        msg.info.width = W
        msg.info.height = H
        msg.info.origin.position.x = X0 - SPAWN_X
        msg.info.origin.position.y = Y0 - SPAWN_Y
        msg.info.origin.orientation.w = 1.0
        msg.data = g.flatten().tolist()
        self.pub_map.publish(msg)

        if self.markers_ver != self.dyn_ver:
            self.markers_ver = self.dyn_ver
            self.publish_obstacles()

        t = self.now()
        if t - self.last_log > 5.0 and self.targets is not None:
            self.last_log = t
            self.get_logger().info(
                f'coverage {self.pct:5.1f}% | phase {self.phase} | '
                f'target {min(self.t_idx, len(self.targets))}/{len(self.targets)}')


    def publish_obstacles(self):
        """Obstacle logger: cluster the unmapped hits and show them in RViz."""
        obs = find_obstacles(self.dyn_raw, self.dyn_first)
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp_now = self.get_clock().now().to_msg()
        for o in obs:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp_now
            m.ns = 'found_obstacles'
            m.id = o['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = o['x'] - SPAWN_X
            m.pose.position.y = o['y'] - SPAWN_Y
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = max(o['width'], 0.15)
            m.scale.y = max(o['depth'], 0.15)
            m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 0.85
            arr.markers.append(m)
            tx = Marker()
            tx.header.frame_id = 'map'
            tx.header.stamp = stamp_now
            tx.ns = 'found_obstacle_labels'
            tx.id = 1000 + o['id']
            tx.type = Marker.TEXT_VIEW_FACING
            tx.action = Marker.ADD
            tx.pose.position.x = o['x'] - SPAWN_X
            tx.pose.position.y = o['y'] - SPAWN_Y
            tx.pose.position.z = 0.45
            tx.pose.orientation.w = 1.0
            tx.scale.z = 0.22
            tx.color.r = tx.color.g = tx.color.b = tx.color.a = 1.0
            tx.text = f"#{o['id']}"
            arr.markers.append(tx)
        self.pub_obst.publish(arr)
        return obs

    def finish(self):
        """Write report.md / report.json / report.png once the run is done."""
        if self.report_done:
            return
        self.report_done = True
        try:
            self.report()
            obs = self.publish_obstacles()
            t0 = self.t_start if self.t_start is not None else self.now()
            for o in obs:
                o['first_seen'] = round(o['first_seen'] - t0, 1)
                for k in ('x', 'y', 'width', 'depth'):
                    o[k] = round(o[k], 3)
            truth, false_pos = match_truth(obs)
            sim_dur = self.now() - t0
            stats = dict(
                generated=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                sim_duration_s=round(sim_dur, 1),
                wall_duration_s=round(time.time() - self.wall_start, 1),
                distance_m=round(self.dist, 1),
                avg_speed_mps=round(self.dist / max(sim_dur, 1e-6), 3),
                coverage_final=round(self.pct, 1),
                coverage_after_lanes=round(self.pct_lanes, 1),
                cleanup_goals=self.cleanup_count,
                replans=self.n_replans,
                stuck_events=self.n_stuck,
                unmapped_obstacles=len(obs))
            base = os.path.expanduser(self.get_parameter('report_dir').value)
            outdir = os.path.join(base, datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S'))
            files = write_report(outdir, stats, obs, truth, false_pos, self.raw, self.visited,
                                 self.coverable & ~self.dyn_excl, self.trace)
            found = sum(1 for t in truth if t['found'])
            self.get_logger().info(
                f'REPORT saved in {outdir} ({", ".join(files)}) | '
                f'{len(obs)} unmapped obstacles, {found}/{len(truth)} known hidden objects found')
        except Exception as e:      # never crash the node because of the report
            self.get_logger().error(f'Report failed: {e}')


def main():
    rclpy.init()
    node = CoveragePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()