from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2400, 2050
BG = "#ffffff"
INK = "#34343b"
PURPLE = "#eeeafd"
PURPLE_EDGE = "#9b8de2"
START_END = "#d8cdfb"
DECISION = "#f0ecff"
MUTED = "#686671"


def font(size: int, bold: bool = False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size)


F_TITLE = font(46, True)
F_NODE = font(30)
F_NODE_BOLD = font(32, True)
F_SMALL = font(24)
F_TINY = font(21)


image = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(image)


def centered_text(box, text, fnt, fill=INK, spacing=8):
    x1, y1, x2, y2 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(((x1 + x2 - w) / 2, (y1 + y2 - h) / 2), text,
                        font=fnt, fill=fill, spacing=spacing, align="center")


def rounded(box, text, fill=PURPLE, outline=PURPLE_EDGE, radius=24,
            fnt=F_NODE, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    centered_text(box, text, fnt)


def ellipse(box, text, fill=START_END, outline=PURPLE_EDGE, fnt=F_NODE):
    draw.ellipse(box, fill=fill, outline=outline, width=3)
    centered_text(box, text, fnt)


def diamond(cx, cy, w, h, text):
    points = [(cx, cy - h // 2), (cx + w // 2, cy),
              (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(points, fill=DECISION, outline=PURPLE_EDGE)
    draw.line(points + [points[0]], fill=PURPLE_EDGE, width=3, joint="curve")
    centered_text((cx - w // 2 + 28, cy - h // 2 + 20,
                   cx + w // 2 - 28, cy + h // 2 - 20), text, F_SMALL)


def arrow(points, label=None, label_at=None, label_side="above"):
    draw.line(points, fill=INK, width=4, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    import math
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


def section(box, label):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=28, fill="#fbfaff", outline="#d6cff4", width=2)
    draw.text((x1 + 26, y1 + 18), label, font=F_NODE_BOLD, fill="#6457a3")


title = "Interactive LangGraph Chatbot — Program Flow"
bbox = draw.textbbox((0, 0), title, font=F_TITLE)
draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, 24), title, font=F_TITLE, fill=INK)

# Main CLI flow
ellipse((940, 110, 1460, 210), "__start__")
rounded((820, 280, 1580, 390), "run_cli()\nread user input")
diamond(1200, 530, 560, 220, "input is\nexit / quit / q?")
rounded((250, 485, 760, 595), "stop all active\nscheduled jobs")
ellipse((250, 700, 760, 800), "__end__")
rounded((820, 710, 1580, 840), "create_search_plan(prompt)\nLLM planner or fallback parser")
diamond(1200, 1010, 560, 220, "plan.should_schedule?")

arrow([(1200, 210), (1200, 280)])
arrow([(1200, 390), (1200, 420)])
arrow([(920, 530), (760, 530)], "yes", (800, 500))
arrow([(505, 595), (505, 700)])
arrow([(1200, 640), (1200, 710)], "no", (1220, 655), "below")
arrow([(1200, 840), (1200, 900)])

# One-off lane
section((70, 1170, 1090, 1900), "ONE-OFF REQUEST")
rounded((170, 1250, 990, 1365), "chatbot.invoke()\nsearch instructions + user prompt")
rounded((135, 1430, 1025, 1660),
        "LangGraph execution\n\nchat_node: LLM response / tool request\n"
        "tools_condition → ToolNode → chat_node\n"
        "Tools: web search · calculator · time · stock · RAG · wait",
        fnt=F_SMALL)
rounded((230, 1740, 930, 1850), "print Assistant response")

arrow([(920, 1010), (650, 1010), (650, 1250)], "no", (735, 980))
arrow([(580, 1365), (580, 1430)])
arrow([(580, 1660), (580, 1740)])
arrow([(230, 1795), (50, 1795), (50, 240), (820, 240), (820, 280)],
      "next prompt", (65, 1770))

# Background scheduling lane
section((1300, 1170, 2330, 1900), "BACKGROUND SCHEDULE")
rounded((1400, 1250, 2230, 1365), "run_scheduled_search()\nstart daemon worker thread")
rounded((1400, 1430, 2230, 1535), "worker: resolve plan + validate\ninterval and run count")
rounded((1400, 1600, 2230, 1705), "run chatbot.invoke(scheduled)\nperform task immediately")
diamond(1815, 1810, 500, 180, "more runs?")
rounded((1360, 1970, 1810, 2040), "wait interval\n(stop_event can cancel)", fnt=F_TINY)
rounded((1900, 1970, 2270, 2040), "done_event.set()\njob complete", fnt=F_TINY)

arrow([(1480, 1010), (1815, 1010), (1815, 1250)], "yes", (1530, 980))
arrow([(1815, 1365), (1815, 1430)])
arrow([(1815, 1535), (1815, 1600)])
arrow([(1815, 1705), (1815, 1720)])
arrow([(1565, 1810), (1360, 1810), (1360, 1970)], "yes", (1400, 1778))
arrow([(1360, 1970), (1360, 1580), (1400, 1580)], "repeat", (1370, 1860))
arrow([(2065, 1810), (2270, 1810), (2270, 1970)], "no", (2100, 1778))

# RAG detail callout
section((70, 1915, 1260, 2040), "RAG TOOL DETAIL")
draw.text((105, 1980),
          "get_rag_chunks → user-uploaded PDF's Chroma index\n"
          "→ similarity search top 4 → return matching chunks",
          font=F_TINY, fill=INK)

image.save("program_flowchart.png", optimize=True)
