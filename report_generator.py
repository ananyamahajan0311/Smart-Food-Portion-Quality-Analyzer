from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

from reportlab.platypus import Image as RLImage
from reportlab.platypus import Spacer
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle

from reportlab.platypus import Image as RLImage
from reportlab.lib.units import inch


def generate_report(original_path, segmented_path, portion, quality, confidence, all_probs):

    doc = SimpleDocTemplate("Food_Analysis_Report.pdf")
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Smart Food Portion & Quality Analyzer Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))

    # Add images
    elements.append(Paragraph("<b>Original Image:</b>", styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(RLImage(original_path, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>Segmented Image:</b>", styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(RLImage(segmented_path, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.5 * inch))

    # Add results table
    data = [
        ["Portion (%)", f"{round(portion,2)}%"],
        ["Quality", quality],
        ["Confidence (%)", f"{round(confidence*100,2)}%"]
    ]

    for cls, prob in all_probs.items():
        data.append([cls, f"{round(prob*100,2)}%"])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(table)

    doc.build(elements)

    print("PDF Report Generated Successfully")