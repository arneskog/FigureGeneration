"""
Template for a new example: one scenario, drawn without and with friction.

Copy this file, change the SCENARIO section, and run from the repository root:
    uv run src/figuregeneration/my_example.py
It prints the margins and writes figures/my_example.tex.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrench3d import Scenario, View, figure, panel  # noqa: E402

MU = 0.3
HF = "\\mathcal{P}^{\\mathrm{HF}}"
PWOF = "\\mathcal{P}^{\\mathrm{PwoF}}"

# --- SCENARIO: change this part ---------------------------------------------
# A contact is (point, normal): the point on the body and the unit normal
# pointing INTO the body, both in the body frame. The normal is normalised
# for you, so (0, 1) and (0, 5) mean the same thing.
#
# Example: equilateral triangle with corners on the unit circle, one peg on
# each edge, shifted off the edge midpoint so the pegs also give torque.
corners = np.array([[np.cos(a), np.sin(a)] for a in np.radians([90, 210, 330])])
contacts = []
for k in range(3):
    a, b = corners[k], corners[(k + 1) % 3]
    mid = (a + b) / 2
    along = (b - a) / np.linalg.norm(b - a)
    inward = -mid / np.linalg.norm(mid)          # edge normal, pointing to the centre
    contacts.append((mid + 0.3 * along, inward))

sc = Scenario("triangle, three pegs", contacts, mu=MU,
              L=1.0,          # torque is divided by L
              p0=(0, 0),      # point torque is taken about
              body={"polygon": corners.tolist()})   # or {"circle": radius}
# -----------------------------------------------------------------------------

VIEW = View(-40.0, 28.0)          # azimuth, elevation in degrees: rotate the 3D plot
OPTS = dict(mark=True, split=True)


def write_figure(path="figures/my_example.tex"):
    m = sc.margins()
    rows = [[
        panel(sc, False, VIEW, f"{sc.name}, without friction: $d({PWOF}) = {m['d_0']:.2f}$.",
              "fig:my-pwof", **OPTS),
        panel(sc, True, VIEW, f"{sc.name}, with friction ($\\mu = {MU}$): $d({HF}) = {m['d_hf']:.2f}$.",
              "fig:my-hf", **OPTS),
    ]]
    figure(rows, path, header=f"% {sc.name}, mu = {MU}.")


if __name__ == "__main__":
    m = sc.margins()
    print(f"{sc.name:20s} d(P^PwoF) = {m['d_0']:+.3f}   d(P^HF) = {m['d_hf']:+.3f}   M (kappa=1) = {m['M']:+.3f}")
    write_figure()
