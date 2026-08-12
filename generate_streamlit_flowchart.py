"""Generate a flowchart for the Streamlit UI in ``streamlit_app.py``."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2600, 2380
BG = "#ffffff"
INK = "#34343b"
PURPLE = "#eeeafd"
PURPLE_EDGE = "#9b8de2"
START_END = "#d8cdfb"
DECISION = "#f0ecff"
MUTED = "#686671"
LANE = "#fbfaff"
LANE_EDGE = "#d6cff4"


def font(size: int, bold: bool = False):
    path = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    return ImageFont.truetype(path, size)


F_TITLE = font(48, True)
F_NODE = font(30)
F_NODE_BOLD = font(30, True)
F_SMALL = font(24)
F_TINY = font(21)

image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)


def centered_text(box, text, fnt, fill=INK, spacing=8):
    x1, y1, x2, y2 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((x1 + x2 - w) / 2, (y1 + y2 - h) / 2),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rounded(box, text, fill=PURPLE, outline=PURPLE_EDGE, radius=24, fnt=F_NODE):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)
    centered_text(box, text, fnt)


def ellipse(box, text, fill=START_END, outline=PURPLE_EDGE, fnt=F_NODE):
    draw.ellipse(box, fill=fill, outline=outline, width=3)
    centered_text(box, text, fnt)


def diamond(cx, cy, w, h, text):
    points = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(points, fill=DECISION, outline=PURPLE_EDGE)
    draw.line(points + [points[0]], fill=PURPLE_EDGE, width=3, joint="curve")
    centered_text(
        (cx - w // 2 + 34, cy - h // 2 + 22, cx + w // 2 - 34, cy + h // 2 - 22),
        text,
        F_SMALL,
    )


def arrow(points, label=None, label_at=None, label_side="above"):
    draw.line(points, fill=INK, width=4, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 18
    left = (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48))
    right = (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48))
    draw.polygon([(x2, y2), left, right], fill=INK)
    if label and label_at:
        bbox = draw.textbbox((0, 0), label, font=F_SMALL)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        lx, ly = label_at
        if label_side == "above":
            ly -= lh + 8
        draw.rectangle((lx - 7, ly - 3, lx + lw + 7, ly + lh + 3), fill=BG)
        draw.text((lx, ly), label, font=F_SMALL, fill=MUTED)


def lane(box, label):
    draw.rounded_rectangle(box, radius=28, fill=LANE, outline=LANE_EDGE, width=2)
    x1, y1, _, _ = box
    draw.text((x1 + 26, y1 + 18), label, font=F_NODE_BOLD, fill="#6457a3")


title = "Streamlit Chat UI — Program Flow"
bbox = draw.textbbox((0, 0), title, font=F_TITLE)
draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, 24), title, font=F_TITLE, fill=INK)

# Shared Streamlit lifecycle.
ellipse((1000, 105, 1600, 205), "__start__")
rounded((850, 275, 1750, 405), "Streamlit reruns streamlit_app.py\n(page load, input, button, or poll)")
rounded((850, 475, 1750, 605), "initialize session state\nmessages · seen_runs · known_jobs")
rounded((850, 675, 1750, 805), "resume persisted schedules once\nload CSS · configure page · inspect active jobs")
diamond(1300, 965, 700, 230, "active schedules or\n15-second grace window?")
rounded((1770, 880, 2410, 1050), "st_autorefresh(interval=3000)\ntrigger the next full rerun")
rounded((850, 1135, 1750, 1265), "render header, sidebar, chat history\nand active-schedule status chips")
diamond(1300, 1435, 700, 230, "new chat prompt?")

arrow([(1300, 205), (1300, 275)])
arrow([(1300, 405), (1300, 475)])
arrow([(1300, 605), (1300, 675)])
arrow([(1300, 805), (1300, 850)])
arrow([(1650, 965), (1770, 965)], "yes", (1680, 935))
arrow([(2090, 1050), (2090, 1090), (1800, 1090), (1800, 1140)], "poll", (2110, 1080))
arrow([(1300, 1080), (1300, 1135)], "no", (1320, 1095), "below")
arrow([(850, 1200), (760, 1200), (760, 1370), (735, 1370), (735, 1465)], "check", (770, 1340))

# Background polling lane: completed run files re-enter the normal render flow.
lane((70, 1320, 810, 2070), "BACKGROUND POLLING")
rounded((145, 1410, 735, 1525), "_fetch_new_scheduled_runs()\nscan .hakathon/history/")
rounded((145, 1600, 735, 1715), "skip seen run files\nread new JSON results")
rounded((115, 1790, 765, 1935), "_render_scheduled_run_card()\nshow completed result as an amber card\nappend it to session chat history", fnt=F_SMALL)
rounded((205, 1995, 675, 2040), "rerun after next poll", fnt=F_TINY)

arrow([(440, 1525), (440, 1600)])
arrow([(440, 1715), (440, 1790)])
arrow([(440, 1935), (440, 1995)])
arrow([(765, 1860), (840, 1860), (840, 1435), (950, 1435)], "continue", (850, 1570))

# Chat request lane: planner decides between the one-off and scheduled paths.
lane((850, 1670, 1750, 2310), "CHAT REQUEST")
rounded((940, 1760, 1660, 1875), "create_search_plan(prompt)\nLLM planner or fallback parser")
diamond(1300, 2045, 620, 210, "plan.should_schedule?")
rounded((900, 2170, 1250, 2275), "_run_chatbot()\nchatbot.invoke()", fnt=F_SMALL)
rounded((1350, 2170, 1700, 2275), "scheduler.start()\nbackground job", fnt=F_SMALL)

arrow([(1300, 1545), (1300, 1670), (1300, 1760)], "yes", (1320, 1620), "below")
arrow([(1300, 1875), (1300, 1940)])
arrow([(1010, 2045), (900, 2045), (900, 2170)], "no", (925, 2015))
arrow([(1590, 2045), (1700, 2045), (1700, 2170)], "yes", (1605, 2015))

# Details below the two request outcomes.
lane((70, 2110, 810, 2310), "ONE-OFF EXECUTION")
rounded((120, 2180, 760, 2270), "LangGraph loop: LLM ↔ tools\nweb · calculator · time · stocks · RAG", fnt=F_TINY)
lane((1830, 1320, 2530, 2310), "SCHEDULED EXECUTION")
rounded((1900, 1410, 2460, 1525), "daemon worker\nresolve + validate plan")
rounded((1900, 1600, 2460, 1715), "run chatbot.invoke()\nwrite run-NNN.json")
diamond(2180, 1845, 500, 180, "more runs?")
rounded((1880, 1960, 2180, 2050), "wait interval\n(stop can cancel)", fnt=F_TINY)
rounded((2210, 1960, 2490, 2050), "done_event.set()\njob complete", fnt=F_TINY)
rounded((1900, 2135, 2460, 2250), "auto-refresh sees the new file\n→ card appears in chat", fnt=F_SMALL)

arrow([(1700, 2222), (1830, 2222), (1830, 1470), (1900, 1470)], "start", (1740, 2190))
arrow([(2180, 1525), (2180, 1600)])
arrow([(2180, 1715), (2180, 1755)])
arrow([(1930, 1845), (1880, 1845), (1880, 1960)], "yes", (1885, 1815))
arrow([(1880, 1960), (1880, 1565), (1900, 1565)], "repeat", (1890, 1880))
arrow([(2430, 1845), (2490, 1845), (2490, 1960)], "no", (2440, 1815))
arrow([(2350, 2050), (2350, 2135)])

# No-prompt path: Streamlit finishes this run and waits for the next event.
rounded((70, 1070, 520, 1175), "no prompt\nwait for next rerun", fnt=F_SMALL)
arrow([(950, 1435), (700, 1435), (700, 1120), (520, 1120)], "no", (720, 1405))

output = Path(__file__).with_name("streamlit_flowchart.png")
image.save(output, optimize=True)
print(output)
