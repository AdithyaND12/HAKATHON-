"""Generate `prompt_to_answer_flowchart.png` — the full prompt → answer flow.

Same visual style as the repo's original flowchart scripts (PIL, white
background, purple rounded boxes, diamonds for decisions, section lanes).
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2600, 2620
BG = "#ffffff"
INK = "#34343b"
PURPLE = "#eeeafd"
PURPLE_EDGE = "#9b8de2"
START_END = "#d8cdfb"
DECISION = "#f0ecff"
MUTED = "#686671"
LANE = "#fbfaff"
LANE_EDGE = "#d6cff4"
LANE_TITLE = "#6457a3"


def font(size: int, bold: bool = False):
    path = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf"
    )
    return ImageFont.truetype(path, size)


F_TITLE = font(46, True)
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
    draw.text((x1 + 26, y1 + 18), label, font=F_NODE_BOLD, fill=LANE_TITLE)


title = "HAKATHON — Prompt to Answer · Program Flow"
bbox = draw.textbbox((0, 0), title, font=F_TITLE)
draw.text(((WIDTH - (bbox[2] - bbox[0])) / 2, 24), title, font=F_TITLE, fill=INK)

# ---- Main spine --------------------------------------------------------------

ellipse((1060, 90, 1540, 185), "user prompt")
rounded((870, 235, 1730, 365), "ENTRY — streamlit_app.py · app.py (CLI) · api.py\nprompt read from chat input / terminal / HTTP")
rounded((870, 415, 1730, 585),
        "create_search_plan(prompt)\n"
        "LLM structured output (function_calling → json_schema → json_mode,\n"
        "retry 429/5xx) · regex fallback · cheap prompts skip the LLM")
diamond(1300, 730, 620, 180, "plan.should_schedule?")

arrow([(1300, 185), (1300, 235)])
arrow([(1300, 365), (1300, 415)])
arrow([(1300, 585), (1300, 640)])
arrow([(990, 730), (600, 730), (600, 1010)], "no", (800, 700))
arrow([(1610, 730), (1810, 730), (1810, 1010)], "yes", (1660, 700))

# ---- One-off lane ------------------------------------------------------------

lane((70, 940, 1130, 2100), "ONE-OFF REQUEST — interactive chat")
rounded((150, 1030, 1050, 1160),
        "build LLM messages: RAG source label (if active doc)\n"
        "search instruction · trimmed history (~20 msgs / 24k chars)\n"
        "user prompt")
rounded((150, 1220, 1050, 1545),
        "LangGraph agent — chat_node ↔ tools\n\n"
        "Gemini LLM · history trimmed to 10k tokens · recursion limit 10\n\n"
        "if tool requested → ToolNode:\n"
        "  web_search (DuckDuckGo) · get_stock_price (Alpha Vantage)\n"
        "  calculator · get_rag_chunks (active PDF) · get_time\n"
        "→ loop back to chat_node until a final answer",
        fnt=F_SMALL)
rounded((220, 1605, 980, 1745),
        "answer: stream tokens to UI · in/out token counts\n"
        "persist chat → conversations.json · LLM auto-title")

arrow([(600, 1160), (600, 1220)])
arrow([(600, 1545), (600, 1605)])
arrow([(600, 1745), (40, 1745), (40, 300), (870, 300)],
      "next prompt", (55, 1720))

# ---- Schedule lane -----------------------------------------------------------

lane((1170, 940, 2530, 2320), "BACKGROUND SCHEDULE — recurring jobs")
rounded((1260, 1030, 2350, 1150),
        "scheduler.start(prompt, interval, run_count, …)\n"
        "validate · ScheduledSearchJob id (8 hex) · → jobs.json snapshot\n"
        "daemon worker thread")
rounded((1260, 1210, 2350, 1330),
        "wait until absolute start (e.g. 3pm, in 5 min) if set\n"
        "pause → hold loop · cancel → stop · restart resumes via jobs.json")
rounded((1260, 1390, 2350, 1515),
        "per run: build messages by task_type — search · reminder ·\n"
        "calculation · rag · chat  (schedule phrases stripped)\n"
        "so the LLM just performs the task once", fnt=F_SMALL)
rounded((1260, 1575, 2350, 1695),
        "chatbot.invoke() — retry ×3 with exponential backoff\n"
        "console print locked · writes history/<id>/run-NNN.json")
diamond(1805, 1845, 520, 180, "more runs?")
rounded((1330, 1985, 1970, 2090), "wait interval\n(pause / cancel honored)", fnt=F_TINY)
rounded((2090, 1985, 2480, 2090), "done_event.set()\njob completed", fnt=F_TINY)
rounded((1330, 2140, 2480, 2255),
        "UI: background poller (@fragment, 3s) sees the new run file\n"
        "→ amber “scheduled run” card lands in the chat that started it",
        fnt=F_SMALL)
rounded((1400, 2290, 2410, 2290), "saved", fnt=F_SMALL, fill=BG, outline=BG)

arrow([(1810, 1150), (1810, 1210)])
arrow([(1810, 1330), (1810, 1390)])
arrow([(1810, 1515), (1810, 1575)])
arrow([(1810, 1695), (1810, 1755)])
arrow([(1545, 1845), (1470, 1845), (1470, 1985)], "yes", (1430, 1815))
arrow([(1380, 1985), (1380, 1900), (1220, 1900), (1220, 1660), (1260, 1660)],
      "repeat", (1230, 1870))
arrow([(2065, 1845), (2210, 1845), (2210, 1985)], "no", (2075, 1815))
arrow([(1400, 2197), (1100, 2197), (1100, 1830), (700, 1830), (700, 1745)],
      "rendered in chat", (1115, 1805))

# ---- RAG lane -----------------------------------------------------------------

lane((70, 2340, 2530, 2570), "RAG — USER-UPLOADED PDFs · ragtool.py")
rounded((110, 2400, 640, 2500),
        "upload → sha256 · index_pdf()\nPyMuPDF text · OCR fallback", fnt=F_TINY)
rounded((700, 2400, 1230, 2500),
        "heading-aware chunks (500/50)\ncontext tag [doc > section]", fnt=F_TINY)
rounded((1290, 2400, 1820, 2500),
        "Jina embeddings · batch 100\nretry/backoff · fail-fast on quota", fnt=F_TINY)
rounded((1880, 2400, 2470, 2500),
        "Chroma rag_<sha16> · marker file\nMMR diverse top-4 → get_rag_chunks", fnt=F_TINY)

arrow([(640, 2450), (700, 2450)])
arrow([(1230, 2450), (1290, 2450)])
arrow([(1820, 2450), (1880, 2450)])

output = Path(__file__).with_name("prompt_to_answer_flowchart.png")
image.save(output, optimize=True)
print(output)