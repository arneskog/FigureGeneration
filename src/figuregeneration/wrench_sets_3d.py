"""
Figure 3 of Sec. 4.5: unit wrench sets P in (f_x, f_y, tau/L) for three cases,
without and with friction, and their signed ball margins d(P).

  ball, pegs 2 and 3 : no closure
  ball, three pegs   : frictional form closure only (L = R, torque about O)
  block, four pegs   : form closure as well (square of half-width 1, L = 1)

Prints the numbers used in the text and writes figures/wrench_sets_3d.tex.
Run from the repository root:  python scripts/wrench_sets_3d.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrench3d import Scenario, View, figure, panel  # noqa: E402

MU = 0.3      # same as \mufric in main.tex

# --- the three cases --------------------------------------------------------
_th = np.radians([90, 210, 330])
_ball = [(np.array([np.cos(a), np.sin(a)]), -np.array([np.cos(a), np.sin(a)])) for a in _th]
_block = [((-0.5, 1), (0, -1)), ((0.5, -1), (0, 1)), ((-1, 0.5), (1, 0)), ((1, -0.5), (-1, 0))]

two = Scenario("ball, pegs 2 and 3", _ball[1:], mu=MU, body={"circle": 1.0})
ball = Scenario("ball, three pegs", _ball, mu=MU, body={"circle": 1.0})
block = Scenario("block, four pegs", _block, mu=MU,
                 body={"polygon": [(-1, -1), (1, -1), (1, 1), (-1, 1)]})

# --- figure options ---------------------------------------------------------
OPTS = dict(mark=True, split=True)
VIEW = View(-40.0, 28.0)          # default view
VIEW_OUTSIDE = View(20.0, 15.0)   # no-closure row: f_y across the page, |d| seen side on
HF = "\\mathcal{P}^{\\mathrm{HF}}"
PWOF = "\\mathcal{P}^{\\mathrm{PwoF}}"


def write_figure(path="figures/wrench_sets_3d.tex"):
    m2, mb, mk = two.margins(), ball.margins(), block.margins()
    rows = [
        [panel(two, False, VIEW_OUTSIDE, f"Ball, pegs 2 and 3, without friction: a segment, $d({PWOF}) = {m2['d_0']:.2f}$.",
               "fig:w3-two-pwof", **OPTS),
         panel(two, True, VIEW_OUTSIDE, f"Ball, pegs 2 and 3, with friction ($\\mu = {MU}$): $d({HF}) = {m2['d_hf']:.2f}$.",
               "fig:w3-two-hf", **OPTS)],
        [panel(ball, False, VIEW, f"Ball, three pegs, without friction: a flat triangle, $d({PWOF}) = 0$.",
               "fig:w3-ball-pwof", **OPTS),
         panel(ball, True, VIEW, f"Ball, three pegs, with friction ($\\mu = {MU}$): $d({HF}) = {mb['d_hf']:.2f}$.",
               "fig:w3-ball-hf", **OPTS)],
        [panel(block, False, VIEW, f"Block, four pegs, without friction: a tetrahedron, $d({PWOF}) = {mk['d_0']:.2f}$.",
               "fig:w3-block-pwof", **OPTS),
         panel(block, True, VIEW, f"Block, four pegs, with friction ($\\mu = {MU}$): $d({HF}) = {mk['d_hf']:.2f}$.",
               "fig:w3-block-hf", **OPTS)],
    ]
    figure(rows, path, header=f"% Sec. 4.5, mu = {MU}. Blue: P^PwoF, orange: P^HF, red: the wrench of each contact.")


if __name__ == "__main__":
    for sc in (two, ball, block):
        m = sc.margins()
        print(f"{sc.name:20s} d(P^PwoF) = {m['d_0']:+.3f}   d(P^HF) = {m['d_hf']:+.3f}   M (kappa=1) = {m['M']:+.3f}")
    write_figure()
