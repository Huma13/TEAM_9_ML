"""Export report/TECHNICAL_REPORT.md and its result figures as an A4 PDF.

Install Pandoc, lxml, reportlab and Pillow. Run after merging all owners.
Pass an output filename to export again after the two-device evidence is added.
"""
import re
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import html as tree
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate,
                               Spacer, TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "TECHNICAL_REPORT_DRAFT.pdf"
styles = getSampleStyleSheet()
body = ParagraphStyle("Report body", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=5)
cell = ParagraphStyle("Table", parent=body, fontSize=7.2, leading=8.9, spaceAfter=0, splitLongWords=True)
caption = ParagraphStyle("Caption", parent=body, fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#35584f"))
doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=43, bottomMargin=40, leftMargin=40, rightMargin=40,
                        title="PathSnap | Technical report")
story = []
seen = set()
def clean(s):
    return escape(s.replace("→"," to ").replace("≤","<=").replace("≥",">=").replace("Δ","delta ").replace("₂","2").replace("₁","1").replace("∈"," in ").replace("θ","theta").replace("·","·"))
def add_image(path):
    p = (ROOT / path).resolve()
    if p in seen:
        return
    if not p.is_file():
        raise FileNotFoundError(p)
    seen.add(p)
    with PILImage.open(p) as im:
        width, height = im.size
    scale = min((doc.width-8)/width, 277/height)
    image = Image(str(p),width*scale,height*scale)
    title = p.stem.replace("_"," ").capitalize()
    story.append(KeepTogether([image, Paragraph(clean(title),caption),Spacer(1,6)]))

def emit(node):
    name=node.tag
    if name in ("h1","h2","h3","h4"):
        lev=int(name[1])
        style=ParagraphStyle("section",parent=styles["Heading"+str(lev)],textColor=colors.HexColor("#164d42"),
                             fontSize={1:21,2:15,3:11.5,4:10}[lev],leading={1:25,2:18,3:14,4:12}[lev])
        story.append(Paragraph(clean(" ".join(node.text_content().split())),style))
    elif name=="table":
        rows=[[Paragraph(clean(" ".join(td.text_content().split())),cell) for td in tr.xpath("./td|./th")]
              for tr in node.xpath(".//tr")]
        n=len(rows[0])
        weights=[1.1,1.05,1.9,1.0,.8,.8,1,.68,1.1] if n==9 else [1]*n
        table=LongTable(rows,colWidths=[x*(doc.width-10)/sum(weights) for x in weights],repeatRows=1,hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dceae4")),("VALIGN",(0,0),(-1,-1),"TOP"),
              ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f6f4")]),
              ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]))
        story.extend([table,Spacer(1,8)])
    elif name in ("ul","ol"):
        for i,li in enumerate(node.findall("li"),1):
            para(li, str(i)+". " if name=="ol" else "• ")
    elif name in ("p","blockquote","pre"):
        para(node,"")
    elif name=="figure":
        for img in node.xpath(".//img"):add_image(img.get("src"))

def para(node,prefix):
    txt=" ".join(node.text_content().split())
    if txt:story.append(Paragraph(clean(prefix+txt),body))
    for path in re.findall(r'\.\./outputs/[\w./-]+\.png',txt):add_image(path)
    for img in node.xpath(".//img"):add_image(img.get("src"))

md=(ROOT/"TECHNICAL_REPORT.md").read_text()
html=subprocess.run(["pandoc","-f","gfm","-t","html"],input=md,text=True,capture_output=True,check=True).stdout
for element in tree.fragment_fromstring(html,create_parent=True):
    emit(element)
def footer(canvas, document):
    canvas.setFont("Helvetica",8)
    canvas.drawString(40,23,"PathSnap | Technical report")
    canvas.drawRightString(A4[0]-40,23,str(document.page))
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
