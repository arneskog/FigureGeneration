"""
wrench3d: unit wrench sets P in the planar wrench space (f_x, f_y, tau/L),
their signed ball margins, and TikZ figures of them.

A scenario is a body with point contacts. Each contact is (p_i, n_i): the
contact point and the unit normal pointing INTO the body, in the body frame.

    sc = Scenario("ball, three pegs", contacts, mu=0.3, L=1.0, p0=(0, 0),
                  body={"circle": 1.0})
    sc.margins(kappa=1.0)            # d(P^HF), d(P^PwoF), M
    lines = panel(sc, friction=True, mark=True, split=True,
                  caption="...", label="fig:...")

Figure features (all optional, per panel):
    rays    thin lines from the origin through every primitive wrench: the
            edges of the wrench cone W; P is the slice of W with total normal
            force one
    split   (friction panels) the frictionless set P^PwoF drawn inside P^HF
    mark    the wrench of each contact in red: a point n_i without friction,
            the line s_i^- -- s_i^+ with friction (dotted on the far side)
    ball    dashed circle of radius d when the origin is inside P
    nearest black arrow to the nearest point of P when the origin is outside
    sketch  small picture of the body, its pegs, normals and friction cones

Orthographic projection; faces of P are drawn back to front.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError


# ----------------------------------------------------------------------------
# Wrenches and margins
# ----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    contacts: list                      # [(p_i, n_i)], n_i into the body
    mu: float = 0.3
    L: float = 1.0                      # torque is divided by L
    p0: tuple = (0.0, 0.0)              # torque reference point
    body: dict = field(default_factory=dict)   # {"circle": r} or {"polygon": [...]}

    def wrenches(self, friction=True):
        """Primitive wrenches, one per row: s_i^+, s_i^- (HF) or n_i (PwoF)."""
        mu = self.mu if friction else 0.0
        p0 = np.asarray(self.p0, float)
        W = []
        for p, n in self.contacts:
            p, n = np.asarray(p, float), np.asarray(n, float)
            n = n / np.linalg.norm(n)
            t = np.array([-n[1], n[0]])
            for f in ([n + mu * t, n - mu * t] if mu > 0 else [n]):
                r = p - p0
                W.append([f[0], f[1], (r[0] * f[1] - r[1] * f[0]) / self.L])
        return np.array(W)

    def margins(self, kappa=1.0):
        d_hf = signed_ball_margin(self.wrenches(True))
        d_0 = signed_ball_margin(self.wrenches(False))
        return {"d_hf": d_hf, "d_0": d_0, "M": d_hf + kappa * max(d_0, 0.0)}


def _min_norm(W):
    """lambda* and G lambda*: the point of conv(W) closest to the origin (QP)."""
    Q = W @ W.T
    m = len(W)
    res = minimize(lambda l: l @ Q @ l, np.full(m, 1 / m), jac=lambda l: 2 * Q @ l,
                   bounds=[(0, None)] * m,
                   constraints=[{"type": "eq", "fun": lambda l: l.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    return res.x, W.T @ res.x


def signed_ball_margin(W, tol=1e-10):
    """Definition 4: Q (nearest facet) inside P, minus the distance outside."""
    try:
        hull = ConvexHull(W)
        b = -hull.equations[:, -1]
        if np.all(b > tol):
            return b.min()
    except (QhullError, ValueError):
        pass                                        # flat P: go to the QP
    dist = np.linalg.norm(_min_norm(W)[1])
    return -dist if dist > 1e-7 else 0.0


def nearest_point(W):
    return _min_norm(W)[1]


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------
class View:
    """Orthographic view from azimuth/elevation (degrees)."""

    def __init__(self, az=-40.0, el=28.0):
        az, el = np.radians(az), np.radians(el)
        self.cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        self.right = np.array([-np.sin(az), np.cos(az), 0.0])
        self.up = np.cross(self.cam, self.right)

    def pt(self, x):
        x = np.asarray(x, float)
        return f"({x @ self.right:.3f},{x @ self.up:.3f})"

    def depth(self, x):
        return np.asarray(x, float) @ self.cam


def _faces(W):
    """Facets of conv(W) as ordered vertex loops with outward normals."""
    hull = ConvexHull(W)
    groups = {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        groups.setdefault(tuple(np.round(eq, 6)), set()).update(simplex.tolist())
    out = []
    for key, idx in groups.items():
        nrm = np.array(key[:3])
        V = W[sorted(idx)]
        c = V.mean(0)
        e1 = V[0] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        out.append((V[np.argsort(np.arctan2((V - c) @ e2, (V - c) @ e1))], nrm))
    return out


def _edge_visible(W, i, j, view, tol=1e-7):
    """Is the segment W[i]--W[j] on a face of conv(W) that faces the viewer?"""
    hull = ConvexHull(W)
    for eq in hull.equations:
        on_face = abs(eq[:3] @ W[i] + eq[3]) < tol and abs(eq[:3] @ W[j] + eq[3]) < tol
        if on_face and eq[:3] @ view.cam > 0:
            return True
    return False


def _polytope(W, view, fill, draw, opacity):
    """(back, front) face commands of conv(W), each ordered far to near."""
    back, front = [], []
    for V, nrm in sorted(_faces(W), key=lambda f: view.depth(f[0].mean(0))):
        path = " -- ".join(view.pt(v) for v in V) + " -- cycle"
        style = f"fill={fill}, fill opacity={opacity}, draw={draw}"
        if nrm @ view.cam > 0:
            front.append(f"        \\filldraw[{style}, line width=0.7pt, line join=round] {path};")
        else:
            back.append(f"        \\filldraw[{style}, draw opacity=0.45, densely dashed, very thin] {path};")
    return back, front


def _flat(W, view, fill, draw):
    """A set that is not a 3D body: a point, a segment or a flat polygon."""
    if len(W) == 1:
        return [f"        \\fill[{draw}] {view.pt(W[0])} circle (1.2pt);"]
    if np.linalg.matrix_rank(W[1:] - W[0], tol=1e-9) == 1:
        d = W - W[0]
        e = d[np.argmax(np.linalg.norm(d, axis=1))]
        s = d @ e
        a, b = W[np.argmin(s)], W[np.argmax(s)]
        return [f"        \\draw[{draw}, line width=1.6pt] {view.pt(a)} -- {view.pt(b)};"]
    c = W.mean(0)
    nrm = np.cross(W[1] - W[0], W[2] - W[0])
    e1 = W[0] - c
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm / np.linalg.norm(nrm), e1)
    uv = np.c_[(W - c) @ e1, (W - c) @ e2]
    loop = W[ConvexHull(uv).vertices]
    return [f"        \\filldraw[fill={fill}, fill opacity=0.7, draw={draw}, thick] "
            + " -- ".join(view.pt(v) for v in loop) + " -- cycle;"]


def _axes(view):
    L = [f"        \\draw[w3 axis] {view.pt((0, 0, 0))} -- {view.pt(v)} node[anchor={a}, inner sep=1pt] {{${lab}$}};"
         for v, lab, a in [((1.45, 0, 0), "f_x", "north"), ((0, 1.45, 0), "f_y", "west"),
                           ((0, 0, 1.25), "\\tau/L", "south")]]
    return L


def _sketch(sc, friction, at=(-1.3, 0.95), size=0.3):
    """Small picture of the body and its pegs, scaled to fit `size`."""
    pts = np.array([p for p, _ in sc.contacts], float)
    if "circle" in sc.body:
        ext = sc.body["circle"]
    elif "polygon" in sc.body:
        ext = np.abs(np.array(sc.body["polygon"], float)).max()
    else:
        ext = np.abs(pts).max()
    k = size / ext
    L = [f"        \\begin{{scope}}[shift={{({at[0]},{at[1]})}}]"]
    if "circle" in sc.body:
        L.append(f"            \\draw[thick, fill=white] (0,0) circle ({k * sc.body['circle']:.3f});")
    elif "polygon" in sc.body:
        P = np.array(sc.body["polygon"], float) * k
        L.append("            \\draw[thick, fill=white] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in P) + " -- cycle;")
    for p, n in sc.contacts:
        p, n = np.asarray(p, float) * k, np.asarray(n, float) / np.linalg.norm(n)
        t = np.array([-n[1], n[0]])
        peg = p - 0.085 * n
        L.append(f"            \\fill[fill=gray!60, draw=gray!40!black] ({peg[0]:.3f},{peg[1]:.3f}) circle (0.07);")
        if friction and sc.mu > 0:
            a = p + 0.2 * (n + sc.mu * t) / np.hypot(1, sc.mu)
            b = p + 0.2 * (n - sc.mu * t) / np.hypot(1, sc.mu)
            L.append(f"            \\fill[orange!55, fill opacity=0.8] ({p[0]:.3f},{p[1]:.3f}) -- "
                     f"({a[0]:.3f},{a[1]:.3f}) -- ({b[0]:.3f},{b[1]:.3f}) -- cycle;")
        q = p + 0.15 * n
        L.append(f"            \\draw[->, >={{Stealth[length=1.4mm]}}, blue!70!black] "
                 f"({p[0]:.3f},{p[1]:.3f}) -- ({q[0]:.3f},{q[1]:.3f});")
    L.append("        \\end{scope}")
    return L


# ----------------------------------------------------------------------------
# Panels and figures
# ----------------------------------------------------------------------------
STYLES = r"""\tikzset{w3 axis/.style={->, >={Stealth[length=1.6mm]}, gray!60!black},
        w3 ball/.style={densely dashed, thick, black!80},
        w3 ray/.style={black!55, semithick},
        w3 rayext/.style={black!55, semithick, densely dotted},
        w3 mark/.style={red!85!black, line width=1.5pt, line cap=round},
        w3 markback/.style={red!85!black, line width=1.3pt, line cap=round, dash pattern=on 0pt off 2.6pt},
        w3 markdot/.style={fill=red!85!black, draw=white, line width=0.4pt}}"""


def panel(sc, friction, view=None, caption="", label="", rays=False,
          split=False, mark=False, ball=True, nearest=True, sketch=True, vertex_labels=(),
          sketch_at=(-1.3, 0.95), scale=1.55, width="0.48\\textwidth"):
    """One subfigure: P^HF (friction=True, orange) or P^PwoF (blue)."""
    view = view or View()
    W = sc.wrenches(friction)
    W0 = sc.wrenches(False)
    d = signed_ball_margin(W)
    L = [f"\\begin{{subfigure}}[t]{{{width}}}", "    \\centering",
         f"    \\begin{{tikzpicture}}[scale={scale}, font=\\small]"]
    L += _axes(view)
    if rays:                                               # edges of the wrench cone
        for w in W:
            L.append(f"        \\draw[w3 ray] {view.pt((0, 0, 0))} -- {view.pt(w)};")
            L.append(f"        \\draw[w3 rayext] {view.pt(w)} -- {view.pt(1.35 * w)};")
    is_body = np.linalg.matrix_rank(W[1:] - W[0], tol=1e-9) == 3 if len(W) >= 4 else False
    color = ("orange!30", "orange!75!black", 0.22) if friction else ("blue!45", "blue!75!black", 0.5)
    back, front = _polytope(W, view, *color) if is_body else ([], _flat(W, view, color[0], color[1]))
    L += back
    if split and friction:                                 # the frictionless set inside P^HF
        body0 = len(W0) >= 4 and np.linalg.matrix_rank(W0[1:] - W0[0], tol=1e-9) == 3
        if body0:
            b0, f0 = _polytope(W0, view, "blue!50", "blue!75!black", 0.55)
            L += b0 + f0
        else:
            L += _flat(W0, view, "blue!50", "blue!75!black")
    if ball and d > 0:
        L.append(f"        \\draw[w3 ball] (0,0) circle ({d:.3f});")
    L.append("        \\fill (0,0) circle (0.9pt);")
    L += front
    if mark:                                               # contact wrenches in red, on top
        if friction:                                       # each point n_i is now the line s_i^- -- s_i^+
            for i in range(len(W0)):
                seg = f"{view.pt(W[2 * i])} -- {view.pt(W[2 * i + 1])}"
                if is_body and not _edge_visible(W, 2 * i, 2 * i + 1, view):
                    L.append(f"        \\draw[w3 markback] {seg};")                           # far side
                    L.append(f"        \\fill[red!85!black] {view.pt(W0[i])} circle (0.9pt);")
                    continue
                L.append(f"        \\draw[white, line width=2.6pt, line cap=round] {seg};")   # halo
                L.append(f"        \\draw[w3 mark] {seg};")
                L.append(f"        \\fill[w3 markdot] {view.pt(W0[i])} circle (1.3pt);")       # n_i = midpoint
        else:
            for n in W0:
                L.append(f"        \\fill[w3 markdot] {view.pt(n)} circle (1.8pt);")
    if nearest and d < 0:
        x = nearest_point(W)
        L.append(f"        \\draw[->, >={{Stealth[length=2mm]}}, very thick, black] (0,0) -- {view.pt(x)};")
        L.append(f"        \\fill[black] {view.pt(x)} circle (1pt);")
        L.append(f"        \\node[font=\\scriptsize, black, anchor=south east, inner sep=1pt] "
                 f"at (-0.04,0.04) {{$|d| = {abs(d):.2f}$}};")
    for i, anchor in vertex_labels:
        L.append(f"        \\node[blue!60!black, font=\\scriptsize, anchor={anchor}] at {view.pt(W0[i])} "
                 f"{{$\\vect{{n}}_{i + 1}$}};")
    if sketch and sc.body:
        L += _sketch(sc, friction, at=sketch_at)
    L += ["    \\end{tikzpicture}", f"    \\caption{{{caption}}}"]
    if label:
        L.append(f"    \\label{{{label}}}")
    L.append("\\end{subfigure}")
    return L


def figure(rows, path, header=""):
    """Write a TikZ figure body: rows = [[panel lines, panel lines], ...]."""
    L = [f"% Generated by scripts/wrench3d.py. Do not edit by hand.", header, STYLES]
    for r, row in enumerate(rows):
        if r:
            L.append("\\\\[6pt]")
        for c, p in enumerate(row):
            if c:
                L.append("\\hfill")
            L += p
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(l for l in L if l) + "\n")
    print("wrote", path)
