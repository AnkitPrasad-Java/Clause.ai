import ollama
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import textwrap
import traceback

# def get_clause_from_ollama(prompt, model='llama3.2:1b'):
#     try:
#         response = ollama.chat(
#             model=model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         print("✅ Ollama responded.")
#         return response['message']['content']
#     except Exception as e:
#         print("❌ Error from Ollama:", e)
#         traceback.print_exc()
#         return "Error generating clause."

def text_to_pdf(text, output_file):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_JUSTIFY
    import re

    doc = SimpleDocTemplate(output_file, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    story = []
    
    # Title
    title_style = styles['h1']
    title_style.alignment = 1 # Center alignment
    story.append(Paragraph("Legal Clause", title_style))
    story.append(Spacer(1, 24))

    # Body text
    body_style = styles['Justify']
    body_style.fontName = 'Times-Roman'
    body_style.fontSize = 12
    body_style.leading = 14

    # Process the text
    lines = text.split('\n')
    section_counter = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Replace **text** with <b>text</b>
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)

        if line.startswith('<b>') and line.endswith('</b>'):
            # Heading
            heading_text = f"{section_counter}. {line}"
            story.append(Paragraph(heading_text, styles['h2']))
            story.append(Spacer(1, 12))
            section_counter += 1
        else:
            # Paragraph
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 12))

    doc.build(story)

# --- Main ---
if __name__ == '__main__':
    # This block is for testing the PDF generation directly
    # You can add a sample text here to test
    sample_text = "**Sample Clause**\n\n    This is a sample clause for testing purposes. It includes bolded text and multiple paragraphs.\n\n    1.  This is the first item.\n    2.  This is the second item.\n    "
    text_to_pdf(sample_text, "test_generated_clause.pdf")
    print("✅ Test PDF saved as 'test_generated_clause.pdf'")
