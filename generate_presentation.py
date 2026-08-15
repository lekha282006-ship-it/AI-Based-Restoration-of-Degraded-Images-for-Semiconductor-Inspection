from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Slide 1
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = 'AI-Based Restoration of Degraded Images\nfor Semiconductor Inspection'
slide.placeholders[1].text = 'Deep Learning for Semiconductor Defect Image Restoration\n\nFinal Project Presentation'

# Slide 2
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = 'Problem and Objective'
textbox = slide.shapes.add_textbox(Inches(0.6), Inches(1.3), Inches(6.3), Inches(4.5))
text_frame = textbox.text_frame
text_frame.word_wrap = True
points = [
    'Semiconductor inspection images often suffer from noisy, degraded low-resolution inputs.',
    'The goal is to recover cleaner structural details for visual inspection and defect analysis.',
    'A compact CNN is used to denoise and super-resolve in a single forward pass.',
    'The method is designed to be lightweight enough to run on CPU and still provide usable outputs.'
]
for i, p in enumerate(points):
    paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
    paragraph.text = p
    paragraph.font.size = Pt(22)

# Slide 3
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = 'Approach'
textbox = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(4.8), Inches(4.0))
text_frame = textbox.text_frame
items = [
    'Residual CNN architecture',
    'Instance normalization',
    'Pixel-shuffle upscaling head',
    'Single-pass restoration',
    'Output constrained to [0, 1]'
]
for i, item in enumerate(items):
    paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
    paragraph.text = item
    paragraph.font.size = Pt(22)

img_path = Path('results/val_grid_demo.png')
if img_path.exists():
    slide.shapes.add_picture(str(img_path), Inches(6.1), Inches(1.8), Inches(6.2), Inches(3.7))

# Slide 4
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = 'Validation Results'
textbox = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(4.6), Inches(3.8))
text_frame = textbox.text_frame
metrics = [
    'Train loss: 0.4737',
    'Validation PSNR: 9.45 dB',
    'Validation SSIM: 0.0549',
    'LPIPS (subset): 0.6600',
    'OOD check: processed 400 unseen test images'
]
for i, item in enumerate(metrics):
    paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
    paragraph.text = item
    paragraph.font.size = Pt(24)

# Slide 5
slide = prs.slides.add_slide(prs.slide_layouts[5])
slide.shapes.title.text = 'Conclusion'
textbox = slide.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.2), Inches(3.5))
text_frame = textbox.text_frame
points = [
    'A compact restoration pipeline was built and validated on real semiconductor data.',
    'The model can operate on previously unseen inputs without retraining, showing practical generalization.',
    'The repo includes training, inference, evaluation, reproduction steps, and the submission summary.',
    'Future work: stronger GPU training and larger models for more aggressive restoration performance.'
]
for i, p in enumerate(points):
    paragraph = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
    paragraph.text = p
    paragraph.font.size = Pt(22)

out = Path('AI-Based-Restoration-of-Degraded-Images-for-Semiconductor-Inspection.pptx')
prs.save(str(out))
print(f'Created {out.resolve()}')
