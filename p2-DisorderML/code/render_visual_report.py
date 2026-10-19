"""Compile a saved Markdown visual report to PDF without loading ML data.

Uses optional reportlab (available in the bundled document runtime), not the
training environment. Supports the headings, tables, lists and local PNGs used
by the research reports; missing figures stop export rather than becoming links.
"""
import argparse
from html import escape
from pathlib import Path
import re


IMAGE = re.compile(r"!\[([^\]]*)\]\((?:<([^>]+)>|([^)]+))\)")


def render(source):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    source = Path(source).resolve(strict=True)
    target = source.parent / (source.parent.name + ".pdf")
    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, filename in (("Report", "Vera.ttf"), ("Report-Bold", "VeraBd.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "Report"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 6
    styles["BodyText"].allowWidows = 0
    styles["BodyText"].allowOrphans = 0
    for name, size in (("Title", 21), ("Heading1", 14), ("Heading2", 12)):
        styles[name].fontName = "Report-Bold"
        styles[name].fontSize = size
        styles[name].leading = size * 1.3
        styles[name].keepWithNext = True
    caption = ParagraphStyle("FigureCaption", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=8.5)
    cell = ParagraphStyle("TableCell", parent=styles["BodyText"], fontSize=8.5, leading=12)
    doc = SimpleDocTemplate(str(target), pagesize=landscape(A4), leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36, title=source.parent.name,
                            author="Niccolo Forte")

    def inline(value):
        value = escape(value.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "→": "->"})))
        value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)

        def link(match):
            label, address = match.groups()
            if not address.startswith(("https://", "http://")):
                address = (source.parent / address).resolve().as_uri()
            return f'<link href="{escape(address, quote=True)}" color="#26746c">{label}</link>'
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, value)

    story, embedded = [], []
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        image = IMAGE.fullmatch(line)
        if image:
            label, bracketed, plain = image.groups()
            image_path = (source.parent / (bracketed or plain)).resolve(strict=True)
            heading = []
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("Heading1", "Heading2"):
                heading.append(story.pop())
            picture = Image(str(image_path))
            factor = min(doc.width / picture.imageWidth, (doc.height - (95 if heading else 55)) / picture.imageHeight)
            picture.drawWidth, picture.drawHeight = picture.imageWidth * factor, picture.imageHeight * factor
            story += [KeepTogether(heading + [Paragraph(inline(label), caption), picture, Spacer(1, 12)])]
            embedded.append(image_path)
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            values = [[v.strip() for v in row.strip("|").split("|")] for row in rows]
            values = [row for row in values if not all(re.fullmatch(r":?-+:?", value) for value in row)]
            columns = len(values[0])
            if any(len(row) != columns for row in values):
                raise ValueError("Inconsistent Markdown table column count.")
            ratios = [.26, .185, .185, .185, .185] if columns == 5 else [.19, .61, .2] if columns == 3 else [1 / columns] * columns
            table = Table([[Paragraph(inline(v), cell) for v in row] for row in values],
                          colWidths=[doc.width * ratio for ratio in ratios], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4efed")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#26746c")),
            ]))
            story += [table, Spacer(1, 10)]
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(inline(line.lstrip("# ")), styles[{1: "Title", 2: "Heading1"}.get(level, "Heading2")]))
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), styles["BodyText"], bulletText="\u2022"))
        else:
            parts = [line]
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "- ", "![")):
                parts.append(lines[index].strip())
                index += 1
            story.append(Paragraph(inline(" ".join(parts)), styles["BodyText"]))

    def footer(canvas, document):
        canvas.setFont("Report", 7)
        canvas.setFillColor(colors.HexColor("#526574"))
        canvas.drawString(36, 19, source.parent.name.replace("-", " ") + " | compiled visual report | status dated in source")
        canvas.drawRightString(doc.pagesize[0] - 36, 19, f"Page {document.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return target, embedded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing report README.md; no training or data reload.")
    args = parser.parse_args()
    pdf, images = render(args.source)
    print(f"Saved {pdf}; {len(images)} figures embedded.")
