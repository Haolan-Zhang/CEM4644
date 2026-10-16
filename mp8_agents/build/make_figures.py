"""The three concept diagrams of the MP8 notebook (shown in its markdown cells from GitHub).

    python build/make_figures.py

workflow.png  one prompt vs. a workflow of fixed steps
agent.png     how an AI agent calls a tool (the loop), with question P5 as the example
skills.png    a prompt vs. a skill (name, description, body), and when the agent loads it
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
FONT = "DejaVu Sans"
COL = {"data": ("#eef1f5", "#8d99ae"), "model": ("#dbe9f6", "#457b9d"), "code": ("#d8f0ec", "#2a9d8f"), "skill": ("#fde8d4", "#e07b39")}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x0, y0, x1, y1, text, kind="data", size=10, weight="normal", align="center", mono=False):
    face, edge = COL[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12", fc=face, ec=edge, lw=1.6))
    x = (x0 + x1) / 2 if align == "center" else x0 + 0.15
    ax.text(x, (y0 + y1) / 2, text, ha=align, va="center", fontsize=size, weight=weight, linespacing=1.35,
            family="DejaVu Sans Mono" if mono else FONT)


def arrow(ax, p, q, text=None, rad=0.0, color="#333333", offset=(0, 0.14), size=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text, ha="center", va="bottom", fontsize=size, color=color, family=FONT)


def legend(ax, x, y, items):
    for name, kind in items:
        face, edge = COL[kind]
        ax.add_patch(FancyBboxPatch((x, y), 0.28, 0.2, boxstyle="round,pad=0.01,rounding_size=0.05", fc=face, ec=edge, lw=1.2))
        ax.text(x + 0.38, y + 0.1, name, va="center", fontsize=9, family=FONT)
        x += 0.5 + 0.075 * len(name)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=150, facecolor="white"); plt.close(fig)
    print("wrote", OUT / name)


def workflow():
    fig, ax = canvas(11, 4.3)
    ax.text(0.15, 3.25, "One\nprompt", fontsize=12, weight="bold", va="center", family=FONT)
    box(ax, 1.3, 2.75, 2.7, 3.75, "Drawing\n+ 3 questions")
    box(ax, 3.1, 2.75, 9.5, 3.75, "AI model (Gemini)\nreads the drawing and computes every quantity in one reply", "model")
    box(ax, 9.9, 2.75, 10.9, 3.75, "CMUs\ngrout")
    arrow(ax, (2.72, 3.25), (3.08, 3.25)); arrow(ax, (9.52, 3.25), (9.88, 3.25))

    ax.text(0.15, 1.35, "Workflow", fontsize=12, weight="bold", va="center", family=FONT)
    box(ax, 1.3, 0.85, 2.7, 1.85, "Drawing")
    box(ax, 3.1, 0.85, 5.4, 1.85, "Step 1 · AI model\nreads the length\nof every side", "model")
    box(ax, 5.8, 0.85, 7.0, 1.85, "44, 26, 24,\n8, 20, 34 ft", size=9)
    box(ax, 7.4, 0.85, 9.5, 1.85, "Step 2 · Code\nadds the sides and\napplies the rules", "code")
    box(ax, 9.9, 0.85, 10.9, 1.85, "CMUs\ngrout")
    for a, b in ((2.72, 3.08), (5.42, 5.78), (7.02, 7.38), (9.52, 9.88)):
        arrow(ax, (a, 1.35), (b, 1.35))
    ax.text(6.3, 0.55, "The steps and their order are fixed in advance; each step does one small job.", ha="center", fontsize=9.5,
            style="italic", color="#444", family=FONT)
    legend(ax, 1.3, 0.1, [("data", "data"), ("AI model", "model"), ("code", "code")])
    save(fig, "workflow.png")


def agent():
    fig, ax = canvas(11, 5.0)
    box(ax, 0.2, 3.2, 2.7, 4.6, "Question (P5)\nDoes submittal\n03 30 00-02 meet the\nmaximum w/cm?")
    box(ax, 3.4, 2.15, 5.6, 3.35, "AI model\n(the agent)\ndecides the next step", "model", weight="bold")
    box(ax, 6.3, 3.75, 10.8, 4.85, "1 · Tool call: the model asks for a tool (name + inputs)", "model", size=9.5)
    ax.text(8.55, 4.0, "lookup_log(log='submittals', search='03 30 00-02')", ha="center", fontsize=8.5, family="DejaVu Sans Mono")
    box(ax, 8.7, 2.25, 10.8, 3.25, "2 · The notebook\nruns the tool", "code")
    box(ax, 6.3, 0.65, 10.8, 1.75, "3 · Tool output, back to the model\n(the submittal's row of the log)\nExterior slabs and walks | cement 560 | fly ash 100 | water 290 | ...", "code", size=9)
    box(ax, 0.2, 0.65, 2.7, 2.05, "Answer\nFINAL: NO\n(w/cm 0.44 > 0.42)", weight="bold")
    arrow(ax, (2.72, 3.9), (3.38, 3.0))
    arrow(ax, (5.62, 3.15), (6.28, 4.1))
    arrow(ax, (10.0, 3.73), (10.0, 3.27))
    arrow(ax, (10.0, 2.23), (10.0, 1.77))
    arrow(ax, (6.28, 1.2), (5.0, 2.13))
    arrow(ax, (3.9, 2.13), (2.72, 1.4))
    ax.text(6.0, 2.2, "repeat until\nit has enough", fontsize=9, color="#457b9d", style="italic", ha="left", family=FONT)
    ax.text(3.05, 1.6, "enough:\nanswer", fontsize=9, color="#333", ha="left", family=FONT)
    ax.text(5.5, 0.18, "Next rounds of the same loop:  search_spec('exterior slabs walks w/cm') → maximum w/cm 0.42  ·  "
            "calculator('290 / (560 + 100)') → 0.44", ha="center", fontsize=9, color="#333", family=FONT)
    save(fig, "agent.png")


def skills():
    fig, ax = canvas(11, 5.2)
    ax.text(2.4, 4.85, "A prompt", fontsize=13, weight="bold", ha="center", family=FONT)
    box(ax, 0.2, 2.2, 4.6, 4.5, "Use these conventions: total wall length\n= sum of the outside dimensions; one CMU\n"
        "covers 0.889 ft²; grouting at S in on center\nfills 8/S of the cells; ...\n\nHow many CMUs for a 48 ft x 40 ft wall,\n4 ft high, with 6 % waste?",
        size=9.5, align="left")
    ax.text(2.4, 1.55, "Written into the message every time.\nThe model reads all of it, needed or not.", ha="center", fontsize=9.5,
            style="italic", color="#444", family=FONT)

    ax.plot([4.95, 4.95], [0.3, 5.0], color="#ccc", lw=1)
    ax.text(8.0, 4.85, "A skill", fontsize=13, weight="bold", ha="center", family=FONT)
    box(ax, 5.3, 4.0, 10.8, 4.5, "name: cmu-foundation-takeoff", "skill", size=9.5, align="left", mono=True)
    box(ax, 5.3, 3.15, 10.8, 3.9, "description (the agent always sees it): Estimating conventions for\nCMU foundation walls. Use it for any CMU or grout quantity question.",
        "skill", size=9, align="left")
    box(ax, 5.3, 2.3, 10.8, 3.05, "body (loaded only when needed): # CMU foundation takeoff\n- Total wall length = sum of the outside dimensions ...  (6 rules)",
        "skill", size=9, align="left")
    box(ax, 5.3, 0.75, 7.85, 1.65, "How many CMUs for\nthis foundation wall?", size=9.5)
    box(ax, 8.25, 0.75, 10.8, 1.65, "What is the status of\nsubmittal 03 30 00-03?", size=9.5)
    arrow(ax, (6.6, 1.67), (6.6, 2.28))
    ax.text(6.7, 1.95, "matches: loads the body", ha="left", va="center", fontsize=8.5, color="#2a9d8f", family=FONT)
    ax.plot([9.5, 9.5], [1.67, 2.28], color="#aaa", lw=1.4, ls=(0, (3, 3)))
    ax.text(9.6, 1.95, "no match: not loaded", ha="left", fontsize=8.5, color="#999", family=FONT)
    ax.text(8.0, 0.3, "Saved once (in HokieAI: My Skills). The agent reads only names and descriptions,\nand loads a body when a task matches it.",
            ha="center", fontsize=9.5, style="italic", color="#444", family=FONT)
    save(fig, "skills.png")


if __name__ == "__main__":
    workflow(); agent(); skills()
