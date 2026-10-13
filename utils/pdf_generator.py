from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf_report(metadata, cobb_angle, severity_label):
    """Generates a downloadable clinical summary report."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, 750, "ScolioAI Diagnostic Analysis Summary")
    
    p.setFont("Helvetica", 12)
    p.drawString(100, 710, f"Patient ID: {metadata.get('Patient ID')}")
    p.drawString(100, 690, f"Modality: {metadata.get('Modality')}")
    p.drawString(100, 670, f"Pixel Spacing: {metadata.get('Pixel Spacing')}")
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 630, f"Calculated Primary Cobb Angle: {cobb_angle:.1f}°")
    p.drawString(100, 610, f"Clinical Severity Classification: {severity_label}")
    
    p.setFont("Helvetica-Oblique", 9)
    p.drawString(100, 550, "Disclaimer: FDA-cleared decision support tool prototype. Human-in-the-loop review required.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer
