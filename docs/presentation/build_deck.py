"""Builds docs/presentation/one_frame_is_enough.pptx (Romanian, 16:9) from the results in docs/next-runs.md and
experiments/. Numbers already reported in docs/next-runs.md are written here; the detection-interval and comparison
slides read their CSVs, so rerun after those runs change.

Usage (any env with python-pptx + pandas):  python docs/presentation/build_deck.py
"""
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("one_frame_is_enough.pptx")
RES = Path("E:/facultate_sem3/edge_ai/results")  # raw run outputs (gitignored); copies live in experiments/

INK = RGBColor(0x14, 0x1E, 0x2E)       # near-black navy: titles, dark slides
TEXT = RGBColor(0x33, 0x3D, 0x4D)
MUTED = RGBColor(0x6B, 0x75, 0x85)
LINE = RGBColor(0xD5, 0xDA, 0xE1)
TINT = RGBColor(0xF1, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
EXPOSED = RGBColor(0xE0, 0x7A, 0x1F)   # amber: exposure, the thing we fight
SAFE = RGBColor(0x1F, 0x8A, 0x84)      # teal: our method / protected
GREY = RGBColor(0x9A, 0xA3, 0xAF)      # baselines
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N_SLIDES = 12


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    if name:
        s.name = name
    return s


def text(slide, x, y, w, h, runs, size=16, color=TEXT, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=6, name=None):
    """runs: a string, or a list of paragraphs; a paragraph is a string or a list of (text, {bold, color, size})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            r.font.name = o.get("font", font)
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=TEXT):
    tb = text(slide, x, y, w, h, items, size=size, color=color, space_after=10)
    for p in tb.text_frame.paragraphs:  # real bullets (buChar), not a typed character
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.28)))
        pPr.set("indent", str(-Inches(0.28)))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
    return tb


def slide_base(n, title, dark=False, notes=""):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = INK if dark else WHITE
    if title:
        text(s, 0.6, 0.42, 12.1, 0.9, title, size=32, color=WHITE if dark else INK, font=HEAD, bold=True,
             anchor=MSO_ANCHOR.MIDDLE, name="Title")
    text(s, 12.0, 7.0, 0.8, 0.3, f"{n} / {N_SLIDES}", size=10, color=MUTED, align=PP_ALIGN.RIGHT, name="Slide number")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, value, label, color=EXPOSED, size=60, label_color=TEXT):
    text(slide, x, y, w, 1.1, value, size=size, color=color, font=HEAD, bold=True)
    text(slide, x, y + 0.2 + size / 64, w, 0.9, label, size=14, color=label_color)


def bar_chart(slide, x, y, w, h, cats, series, colors, title=None, maxv=100, fmt='0"%"', legend=True,
              chart_type=XL_CHART_TYPE.COLUMN_CLUSTERED, gap=60, label_size=11):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(chart_type, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.name, ch.font.size, ch.font.color.rgb = BODY, Pt(12), TEXT
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        tp = ch.chart_title.text_frame.paragraphs[0]
        tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.color.rgb = Pt(14), True, INK
        tp.runs[0].font.name = BODY
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
        ch.legend.font.size = Pt(12)
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = maxv, 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(11), MUTED
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    ca = ch.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(12), TEXT
    ca.format.line.color.rgb = LINE
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = gap, -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.font.size, dl.font.color.rgb = Pt(label_size), TEXT
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s_, c in zip(plot.series, colors):
        s_.format.fill.solid()
        s_.format.fill.fore_color.rgb = c
    return ch


def table(slide, x, y, w, h, rows, col_w=None, size=13, header_fill=INK, bold_rows=(), first_col_w=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(v)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (TINT if r % 2 == 0 else WHITE)
            for p in cell.text_frame.paragraphs:  # a newline in the value makes a second paragraph
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
                for run in p.runs:
                    f = run.font
                    f.name, f.size = BODY, Pt(size)
                    f.color.rgb, f.bold = (WHITE, True) if r == 0 else (TEXT, r in bold_rows)
    return t


def csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def pct(df, system, col="pct_identities_leaked"):
    if df is None:
        return None
    r = df[df.system == system]
    return float(r[col].iloc[0]) if len(r) else None


def fmt_pct(v):
    return "în curs" if v is None else f"{v:.0f}%"


# ---------------------------------------------------------------- 1. title
s = slide_base(1, None, dark=True, notes=(
    "Ideea într-o frază: dacă fața cuiva apare neanonimizată într-un singur cadru, identitatea lui e compromisă. "
    "Măsurăm expunerea la nivel de persoană și o reducem pe o placă NXP i.MX 8M Plus, în timp real."))
text(s, 0.9, 2.0, 11.5, 1.4, "Un singur cadru e suficient", size=54, color=WHITE, font=HEAD, bold=True)
text(s, 0.9, 3.35, 11.8, 1.2, "Expunerea la nivel de identitate și recuperarea fețelor ratate "
     "pe NXP i.MX 8M Plus", size=22, color=RGBColor(0xC9, 0xD2, 0xDE))
text(s, 0.9, 5.6, 11.0, 0.5, "Edge Computing Applications  ·  Valerian (vedere artificială, intimitate)  ·  "
     "Moro (edge AI, integrare)", size=16, color=GREY)

# ---------------------------------------------------------------- 2. the problem
s = slide_base(2, "Recall-ul pe cadre ascunde câte persoane sunt expuse", notes=(
    "Recall-ul pe cadre e metrica obișnuită. Dar o persoană e expusă dacă scapă măcar o dată. Pe Music Video, "
    "88% din fețe sunt detectate în fiecare cadru, iar totuși 94% dintre persoane apar cel puțin o dată "
    "recognoscibile pentru un atacator cu ArcFace."))
for i, (a, b, la, lb) in enumerate([("88%", "94%", "dintre fețe detectate, cadru cu cadru", "dintre persoane expuse măcar o dată"),
                                    ("94%", "86%", "recall pe cadre, ChokePoint (prag OpenCV 0.9)", "dintre persoane expuse")]):
    y = 1.75 + i * 2.45
    box(s, 0.6, y, 7.4, 2.05, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Stat card {i + 1}")
    stat(s, 0.95, y + 0.2, 3.0, a, la, color=GREY)
    text(s, 3.85, y + 0.35, 0.6, 0.8, "→", size=40, color=MUTED, font=HEAD)
    stat(s, 4.6, y + 0.2, 3.2, b, lb, color=EXPOSED)
text(s, 8.6, 1.85, 4.2, 0.5, "De ce", size=20, color=INK, font=HEAD, bold=True)
bullets(s, 8.6, 2.45, 4.2, 4.2, [
    "Un cadru ratat într-o pistă lungă ajunge: atacatorul are nevoie de o singură față clară.",
    "Ratările se adună la începutul fiecărei apariții, înainte ca detectorul să „prindă” persoana.",
    "Deci măsurăm procentul de persoane expuse, nu procentul de cadre."], size=16)

# ---------------------------------------------------------------- 3. protocol
s = slide_base(3, "Cum măsurăm: un atacator separat de orice detector", notes=(
    "Atacatorul (judge) e ArcFace R50, cu pragul ales la 1% rată de fals-accept pe perechi de impostori. "
    "Toate sistemele folosesc același blur, aceeași scală a măștii 1.3, aceleași videoclipuri."))
steps = [("1", "Acoperire", "O față e protejată dacă masca acoperă cel puțin 80% din caseta ei reală."),
         ("2", "Atac", "Altfel, o decupăm din cadrul anonimizat și ArcFace o compară cu galeria persoanei "
                       "(prag la 1% fals-accept)."),
         ("3", "Identitate", "Persoană expusă = cel puțin un cadru recunoscut. Raportăm și cea mai lungă "
                             "serie de cadre expuse.")]
for i, (n, h, d) in enumerate(steps):
    x = 0.6 + i * 4.15
    box(s, x, 1.7, 3.85, 2.75, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Step {n}")
    c = box(s, x + 0.3, 1.95, 0.7, 0.7, fill=SAFE, shape=MSO_SHAPE.OVAL, name=f"Step {n} badge")
    text(s, x + 0.3, 1.95, 0.7, 0.7, n, size=22, color=WHITE, font=HEAD, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 1.15, 2.05, 2.5, 0.5, h, size=20, color=INK, font=HEAD, bold=True)
    text(s, x + 0.3, 2.85, 3.3, 1.5, d, size=15)
table(s, 0.6, 4.85, 12.1, 1.9, [
    ["Set de date", "Persoane", "Ce testează"],
    ["ChokePoint (test, 24 secvențe)", "29", "supraveghere, mai multe fețe; galerie din alte secvențe"],
    ["Music Video (8 clipuri, 95k fețe)", "50", "schimbări de cadru, mișcare rapidă; nimic reglat pe el"],
    ["Degradări sintetice", "—", "întuneric, blur, JPEG, ocluzie, rezoluție mică, pe trei niveluri"]],
    col_w=[4.2, 1.4, 6.5], size=14)

# ---------------------------------------------------------------- 4. RQ2 recovery
s = slide_base(4, "Tracker + 3 cadre de întârziere recuperează fețele ratate", notes=(
    "window3 e reimplementarea noastră a MBAM (fereastră simetrică). lite+back3 e trackerul nostru numpy de pe "
    "placă plus un buffer de 3 cadre: fiecare cadru e mascat și cu casetele din următoarele 3. Pe ChokePoint val la "
    "pragul 0.9: 83 -> 34% (stock) și 48 -> 10% (fine-tuned), p = 0.0001."))
mv = csv(ROOT / "experiments/musicvideo/results_op.csv")
cats = ["Per cadru", "window3\n(MBAM, reimpl.)", "lite + back3\n(al nostru)", "+ regula de\nîntuneric"]
sfx = ["", "+window3", "+lite+back3", "+risk_d60_t0.1+lite+back3"]
bar_chart(s, 0.5, 1.45, 7.6, 5.4, cats,
          [("YuNet stock", [pct(mv, "yunet_480x640_npu" + x) for x in sfx]),
           ("YuNet fine-tuned", [pct(mv, "yunet_ft_480x640_npu" + x) for x in sfx])],
          [GREY, SAFE], title="Music Video: % persoane expuse (NPU, INT8)")
text(s, 8.6, 1.7, 4.2, 0.5, "Ce face diferența", size=20, color=INK, font=HEAD, bold=True)
bullets(s, 8.6, 2.3, 4.2, 4.5, [
    [("Trackerul păstrează ", {}), ("detecțiile slabe", {"bold": True}), (" ale unei piste existente.", {})],
    [("Bufferul de 3 cadre acoperă ", {}), ("începutul apariției", {"bold": True}),
     (", înainte ca detectorul să tragă.", {})],
    [("Music Video, stock: 94 → 76% (p = 0.004); ChokePoint val la 0.9: 83 → 34% (p = 0.0001).", {})],
    [("Aceeași arie mascată: 1.4% din cadru pe ChokePoint.", {"color": MUTED})]], size=16)

# ---------------------------------------------------------------- 5. RQ3 dark rule
s = slide_base(5, "Ce rămâne e o problemă de prag, nu de detector", notes=(
    "Pe toate condițiile în afară de întuneric, 100% din fețele încă expuse aveau o detecție, 99% cu scor >= 0.3: "
    "detectorul le-a văzut, pragul le-a aruncat. Regula: dacă media de gri a cadrului e sub 60, pragul scade la 0.1. "
    "Pe cadre normale nu se declanșează, deci aria mascată rămâne aceeași. Detectorul de poză a fost scos: ar salva cel "
    "mult o persoană pe condiție."))
box(s, 0.6, 1.6, 4.3, 5.2, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Leak anatomy card")
stat(s, 0.95, 1.85, 3.8, "≥99%", "dintre fețele încă expuse aveau o detecție cu scor ≥ 0.3: detectorul le-a "
     "văzut, pragul le-a aruncat", color=EXPOSED)
text(s, 0.95, 4.15, 3.7, 0.5, "Regula", size=18, color=INK, font=HEAD, bold=True)
text(s, 0.95, 4.65, 3.7, 2.0, [[("media de gri < 60", {"bold": True}), (" → prag 0.1", {})],
                                "Altfel, pragul de operare (0.6 / 0.7).",
                                [("0.8 ms pe cadru pe placă.", {"color": MUTED})]], size=16)
bar_chart(s, 5.3, 1.45, 7.5, 5.4, ["ChokePoint\nstock", "ChokePoint\nfine-tuned", "Music Video\nstock",
                                   "Music Video\nfine-tuned"],
          [("lite + back3", [34.5, 72.4, 76, 84]), ("+ regula de întuneric (T = 0.1)", [10.3, 37.9, 60, 56])],
          [GREY, SAFE], title="Întuneric sintetic dark:2: % persoane expuse", fmt='0"%"')

# ---------------------------------------------------------------- 6. pipeline on the board
s = slide_base(6, "Pe placă: fiecare unitate lucrează la alt cadru", notes=(
    "VPU decodează H.264, GPU-ul 2D (g2d) convertește culoarea și redimensionează la intrarea detectorului, NPU rulează "
    "YuNet INT8 cu intrare uint8, CPU face decodarea ancorelor, NMS, trackerul și blur-ul. Firele lucrează în paralel, "
    "deci debitul e dat de etapa cea mai lentă, nu de suma lor."))
stages = [("VPU + g2d", "decodare H.264,\nculoare, redimensionare", "1.4 ms"),
          ("NPU", "YuNet INT8\n480×640, uint8", "10.4 ms"),
          ("CPU", "ancore + NMS", "3.6 ms"),
          ("CPU", "tracker lite +\nregula de întuneric", "1.0 ms"),
          ("CPU", "look-back 3 +\nfast_blur", "< 3 ms")]
for i, (u, d, ms) in enumerate(stages):
    x = 0.6 + i * 2.5
    hl = u == "NPU"
    box(s, x, 1.9, 2.15, 2.4, fill=SAFE if hl else TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Stage {i + 1}")
    col = WHITE if hl else INK
    text(s, x + 0.2, 2.05, 1.8, 0.45, u, size=18, color=col, font=HEAD, bold=True)
    text(s, x + 0.2, 2.55, 1.8, 1.0, d, size=14, color=WHITE if hl else TEXT)
    text(s, x + 0.2, 3.55, 1.8, 0.5, ms, size=20, color=col, font=HEAD, bold=True)
    if i < len(stages) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.17), Inches(2.95), Inches(0.3), Inches(0.3))
        a.fill.solid()
        a.fill.fore_color.rgb = GREY
        a.line.fill.background()
        a.name = f"Arrow {i + 1}"
for i, (v, l) in enumerate([("89.5 FPS", "un flux, cât de repede se poate\n(configurația finală)"),
                            ("3 × 25 FPS", "camere live, toate la 25 FPS,\nlatență p50 146 ms"),
                            ("40 ms", "bugetul pe cadru pentru tot\npipeline-ul la 25 FPS")]):
    stat(s, 0.6 + i * 4.15, 4.75, 3.9, v, l, color=SAFE if i < 2 else INK, size=40)

# ---------------------------------------------------------------- 7. RQ4 quantization
s = slide_base(7, "INT8 nu costă intimitate măsurabil și e de ~3 ori mai rapid", notes=(
    "Șase rețete de cuantizare simulate pe desktop (onnxruntime, toate operațiile cuantizate, ca onnx2tf), validate "
    "contra detecțiilor reale de pe NPU (p >= 0.75). Nicio rețetă nu e mai bună decât INT8-ul livrat pe niciunul din "
    "cele 4 seturi; nicio diferență față de FP32 nu e semnificativă. int16 ar recupera tendința de pe Music Video, dar "
    "delegatul VX al NPU-ului îl respinge și rulează pe CPU cu 2.2 s pe cadru."))
table(s, 0.6, 1.6, 7.9, 4.3, [
    ["% persoane expuse, lite + back3", "ChokePoint dark:2\nstock / fine-tuned", "Music Video\nstock / fine-tuned"],
    ["FP32", "34 / 62", "76 / 80"],
    ["INT8 livrat (per-tensor, min-max)", "41 / 69", "84 / 86"],
    ["per-channel", "52 / 66", "86 / 88"],
    ["percentilă 99.999%", "34 / 66", "—"],
    ["calibrare pe cadre întunecate", "48 / 52", "—"],
    ["activări int16", "31 / 62", "78 / 80"]],
    col_w=[3.5, 2.2, 2.2], size=14, bold_rows=(2,))
text(s, 0.6, 6.05, 7.9, 0.8, "Simulare desktop a INT8 de pe placă, ChokePoint test (29 persoane) și Music Video (50). "
     "Niciun test pereche nu e semnificativ (p minim 0.125). Calibrarea pe întuneric nu s-a replicat pe val și Music Video.",
     size=12, color=MUTED)
for i, (v, l, c) in enumerate([("21 ms", "INT8 pe NPU (p95)", SAFE), ("54 ms", "FP32 pe CPU-ul plăcii", GREY),
                               ("2.2 s", "int16: respins de NPU,\nrulează pe CPU", EXPOSED)]):
    stat(s, 9.0, 1.55 + i * 1.75, 3.8, v, l, color=c, size=40)

# ---------------------------------------------------------------- 8. detection interval N
s = slide_base(8, "Intervalul de detecție N: cât costă viteza", notes=(
    "Detectorul rulează doar pe fiecare al N-lea cadru; între ele păstrăm casetele ultimului cadru cheie (hold), cu "
    "același tracker și look-back. Pe placă, N = 2 permite 4 camere la 25 FPS."))
sk = {t: csv(RES / f"repro_desktop/skip/{t}.csv") for t in ("clean", "dark2")}
skm = csv(RES / "musicvideo/runs/skip.csv")
pol = "+risk_d60_t0.1+lite+back3"


def skip_sys(d, n):
    return f"{d}{pol}" if n == 1 else f"{d}+skip{n}+risk_d60_t0.1+lite+hold+back3"


Ns = [1, 2, 3, 5]
bar_chart(s, 0.5, 1.45, 7.6, 5.4, [f"N = {n}" for n in Ns],
          [("ChokePoint dark:2", [pct(sk["dark2"], skip_sys("yunet_ft_480x640_npu", n)) for n in Ns]),
           ("Music Video", [pct(skm, skip_sys("yunet_ft_480x640_npu", n)) for n in Ns])],
          [GREY, EXPOSED], title="Fine-tuned, configurația finală: % persoane expuse")
table(s, 8.5, 1.6, 4.3, 2.6, [["N", "FPS maxim\n(un flux)", "4 camere\nFPS / cameră"],
                               ["1", "89.5", "22.5"], ["2", "132", "25.0"], ["3", "138", "25.0"], ["5", "140", "25.0"]],
      col_w=[0.8, 1.75, 1.75], size=14)
text(s, 8.5, 4.45, 4.3, 2.3, [
    [("N = 2 e gratis pe ChokePoint și costă 8 puncte pe Music Video (mișcare rapidă).", {"bold": True})],
    "Peste N = 2 câștigul de viteză e mic: limita devine decodarea și CPU-ul, nu NPU-ul.",
    [("Măsurat pe placă, 30 s pe configurație (experiments/edge/every*.json).", {"color": MUTED, "size": 12})]],
     size=15)

# ---------------------------------------------------------------- 9. real time, cameras
s = slide_base(9, "Configurația finală ține 25 FPS pe trei camere", notes=(
    "Configurația finală: YuNet fine-tuned INT8 pe NPU, trackerul lite, regula de întuneric, look-back 3, fast_blur. "
    "Camerele sunt simulate ca surse live la 25 FPS. La 4 camere NPU-ul e la 92% și cade la 22.5 FPS; cu N = 2 toate "
    "4 țin 25 FPS."))
bar_chart(s, 0.5, 1.45, 7.6, 5.4, ["1 cameră", "2 camere", "3 camere", "4 camere", "4 camere, N = 2"],
          [("FPS pe cameră", [24.99, 24.98, 24.98, 22.47, 24.98])], [SAFE], title="FPS pe cameră (sursă live la 25 FPS)",
          maxv=30, fmt='0.0', legend=False)
table(s, 8.5, 1.6, 4.3, 3.1, [["Camere", "Latență p50", "NPU ocupat"],
                               ["1", "134 ms", "26%"], ["2", "142 ms", "52%"], ["3", "146 ms", "78%"],
                               ["4", "366 ms", "92%"], ["4, N = 2", "139 ms", "52%"]], col_w=[1.4, 1.45, 1.45], size=14)
text(s, 8.5, 4.95, 4.3, 1.8, [
    "Latența include cele 3 cadre de look-back (120 ms la 25 FPS): prețul recuperării.",
    [("Temperatura maximă a SoC-ului: 40 °C.", {"color": MUTED})]], size=15)

# ---------------------------------------------------------------- 10. comparison with other systems
s = slide_base(10, "Recuperarea funcționează pe orice detector", notes=(
    "Toate sistemele: aceeași mască, același blur, același atacator. Detectoarele mari (deface, SCRFD, YOLOv8n) rulează "
    "pe CPU-ul desktopului, EgoBlur (Faster R-CNN, ~104M parametri) pe GPU; doar YuNet INT8 încape în bugetul de 40 ms "
    "al plăcii. Nici detectorul de referință cel mai greu nu scapă de problema unui singur cadru: pe dark:2 are 97% "
    "recall pe cadre și expune 76% dintre persoane; recuperarea îl duce la 7%. La pragul 0.5: 99.3% recall, tot 31% "
    "expuși; Music Video 68 → 36%, la egalitate cu SCRFD-10G."))
oth = {t: csv(RES / f"repro_desktop/others/{t}.csv") for t in ("clean", "dark2")}
othm = csv(RES / "musicvideo/runs/others.csv")
dfc = {t: csv(ROOT / f"experiments/deface/chokepoint_{t}.csv") for t in ("clean", "dark2")}
dfm = csv(ROOT / "experiments/deface/musicvideo.csv")


def cell(tabs, mv_tab, d):
    out = []
    for t in ("clean", "dark2"):
        a, b = pct(tabs[t], d), pct(tabs[t], d + "+lite+back3")
        out.append("în curs" if a is None else f"{a:.0f} → {b:.0f}")
    a, b = pct(mv_tab, d), pct(mv_tab, d + "+lite+back3")
    out.append("în curs" if a is None else f"{a:.0f} → {b:.0f}")
    return out


rows = [["Detector", "ChokePoint clean", "ChokePoint dark:2", "Music Video", "Pe placă?"]]
rows.append(["YuNet stock (NPU INT8)", "0 → 0", "100 → 34", "94 → 76", "da, 11 ms"])
rows.append(["YuNet fine-tuned (NPU INT8)", "0 → 0", "97 → 72", "96 → 80", "da, 11 ms"])
rows.append(["deface (CenterFace)", *cell(dfc, dfm, "deface"), "nu, 61 ms CPU"])
for d, lab in (("scrfd_500m", "SCRFD-500M"), ("scrfd_10g", "SCRFD-10G"), ("yolov8n_face", "YOLOv8n-face")):
    ms = pct(oth["clean"], d, "p50_ms")
    rows.append([lab, *cell(oth, othm, d), "—" if ms is None else f"neîncercat; {ms:.0f} ms CPU"])
ego = {t: csv(ROOT / f"experiments/others/egoblur_{t}.csv") for t in ("clean", "dark2", "musicvideo")}
rows.append(["EgoBlur (Faster R-CNN)", *cell(ego, ego["musicvideo"], "egoblur"),
             f"nu, {pct(ego['clean'], 'egoblur', 'p50_ms'):.0f} ms GPU"])
table(s, 0.6, 1.5, 12.1, 4.2, rows, col_w=[3.4, 2.2, 2.2, 2.2, 2.1], size=14)
text(s, 0.6, 5.85, 12.1, 1.0, [
    [("% persoane expuse, per cadru → cu trackerul lite + 3 cadre de look-back. ", {}),
     ("Același câștig apare și pe detectoare mai mari: metoda nu depinde de detector.", {"bold": True})],
    [("Prag implicit al fiecărui detector (EgoBlur 0.9); deface, SCRFD, YOLO pe CPU desktop, EgoBlur pe GPU RTX 3060.",
      {"color": MUTED, "size": 12})]],
     size=15)

# ---------------------------------------------------------------- 11. answers to the RQs
s = slide_base(11, "Răspunsuri la cele patru întrebări", notes="Rezumatul pe întrebări de cercetare.")
qa = [("RQ1", "Recall vs expunere", "88% recall pe cadre expune 94% dintre persoane (Music Video). Pe ChokePoint 94% "
       "recall expune 86%."),
      ("RQ2", "Recuperare temporală", "Trackerul lite + look-back 3: 94 → 76% (MV), 83 → 34% (ChokePoint val, 0.9), la "
       "aceeași arie mascată. Detectorul de poză nu e necesar."),
      ("RQ3", "Prag adaptiv", "Regula de luminozitate bate orice prag fix: dark:2 34 → 10% (stock), MV dark 84 → 56% "
       "(fine-tuned); pe cadre normale nu schimbă nimic."),
      ("RQ4", "Placa", "INT8 pe NPU fără cost de intimitate măsurabil, 3× mai rapid; 3 camere la 25 FPS, 4 cu N = 2. "
       "int16 nu rulează pe NPU.")]
for i, (q, h, d) in enumerate(qa):
    x, y = 0.6 + (i % 2) * 6.15, 1.6 + (i // 2) * 2.65
    box(s, x, y, 5.9, 2.4, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"{q} card")
    b = box(s, x + 0.3, y + 0.3, 0.95, 0.6, fill=SAFE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"{q} badge")
    text(s, x + 0.3, y + 0.3, 0.95, 0.6, q, size=16, color=WHITE, font=HEAD, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 1.45, y + 0.35, 4.2, 0.5, h, size=18, color=INK, font=HEAD, bold=True)
    text(s, x + 0.3, y + 1.05, 5.3, 1.3, d, size=14)

# ---------------------------------------------------------------- 12. conclusion
s = slide_base(12, "Concluzii și pașii următori", dark=True, notes=(
    "Mesajul principal: măsurați persoane, nu cadre. Un detector mic pe NPU plus recuperare temporală și un prag "
    "adaptiv ajung în buget, iar cuantizarea INT8 nu e veriga slabă."))
LIGHT = RGBColor(0xC9, 0xD2, 0xDE)
text(s, 0.6, 1.6, 6.0, 0.5, "Ce am arătat", size=22, color=WHITE, font=HEAD, bold=True)
bullets(s, 0.6, 2.2, 5.8, 4.5, [
    "Recall-ul pe cadre subestimează sistematic expunerea persoanelor.",
    "Un tracker numpy și 3 cadre de întârziere reduc expunerea pe orice detector.",
    "Fețele scăpate sunt văzute de detector; un prag după luminozitate le recuperează gratis pe cadre normale.",
    "Totul rulează pe i.MX 8M Plus: INT8 pe NPU, 25 FPS pe 3 camere."], size=16, color=LIGHT)
text(s, 7.0, 1.6, 5.7, 0.5, "Ce urmează", size=22, color=WHITE, font=HEAD, bold=True)
bullets(s, 7.0, 2.2, 5.7, 4.5, [
    "Confirmarea regulii de întuneric și a RQ3 în echipă.",
    "Fețe mici: intrare mai mare sau tiling, la cost de NPU.",
    "Întunericul sever (dark:3) cere un detector antrenat pe date întunecate sau o cameră IR.",
    "SCRFD-500M pe NPU; propriile înregistrări, cu acord scris."], size=16, color=LIGHT)

prs.save(OUT)
print(OUT)
