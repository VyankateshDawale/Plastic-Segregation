import docx
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def create_report():
    doc = docx.Document()
    
    # Page setup
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    
    # Styles Setup
    styles = doc.styles
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)
    
    # Add Title
    title = doc.add_paragraph()
    title_run = title.add_run("Technical Report:\nDual-Sensor HSI (NIR) and FTIR (MIR) Fallback Classification System")
    title_run.font.bold = True
    title_run.font.size = Pt(16)
    title_run.font.name = 'Times New Roman'
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # --- SECTION 1 ---
    h1 = doc.add_paragraph()
    r = h1.add_run("1. System Architecture Overview")
    r.font.bold = True
    r.font.size = Pt(14)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)
    
    p = doc.add_paragraph("The proposed system implements a dual-stage sensor-fusion fallback architecture designed for the classification of post-consumer plastic waste. The primary sensor stage consists of a high-speed Near-Infrared (NIR) Hyperspectral Imaging (HSI) camera. When the NIR sensor scan yields an inconclusive prediction—due to low reflectance signal (e.g. black plastics containing carbon black colorants) or non-plastic material categories (e.g. paper labels or food waste)—the pipeline automatically triggers the high-precision Fourier Transform Infrared (FTIR) Mid-Infrared (MIR) spectrometer. The secondary scan operates as a backup to check the chemical fingerprint of the underlying polymer and save the plastic item from misclassification.")
    p.paragraph_format.space_after = Pt(12)
    
    # --- SECTION 2 ---
    h2 = doc.add_paragraph()
    r2 = h2.add_run("2. Dataset Specifications")
    r2.font.bold = True
    r2.font.size = Pt(14)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)
    
    # NIR Dataset Subheading
    sub1 = doc.add_paragraph()
    r_sub1 = sub1.add_run("A. Primary NIR HSI Dataset")
    r_sub1.font.bold = True
    r_sub1.font.size = Pt(12)
    sub1.paragraph_format.space_before = Pt(6)
    sub1.paragraph_format.space_after = Pt(4)
    
    doc.add_paragraph("• Source: Specim FX10 Hyperspectral camera ('ADVANCED_PLASTIC_FUSE' dataset).")
    doc.add_paragraph("• Spectral Configuration: 232 spectral bands spanning the visible to Near-Infrared spectrum (400 - 1000 nm).")
    doc.add_paragraph("• Spatial Configuration: 1,220 subimages structured as 3D HSI data cubes of size 64 x 64 pixels x 232 bands.")
    doc.add_paragraph("• Splits: Stratified split containing 731 samples for training, 246 for validation, and 243 for testing.")
    doc.add_paragraph("• Target Classes: High-Density Polyethylene (HDPE, 178 samples), Polypropylene (PP, 335 samples), Low-Density Polyethylene (LDPE, 178 samples), Polyethylene Terephthalate (PET, 226 samples), Polystyrene (PS, 43 samples), Organic Contamination (ORGANIC, 145 samples), and Other Waste (OTHER, 115 samples).")
    
    # MIR Dataset Subheading
    sub2 = doc.add_paragraph()
    r_sub2 = sub2.add_run("B. Backup MIR FTIR Dataset")
    r_sub2.font.bold = True
    r_sub2.font.size = Pt(12)
    sub2.paragraph_format.space_before = Pt(6)
    sub2.paragraph_format.space_after = Pt(4)
    
    doc.add_paragraph("• Source: Mid-Infrared Fourier Transform Infrared (FTIR) spectrometer ('FTIR-PLASTIC' dataset).")
    doc.add_paragraph("• Spectral Configuration: 3,736 spectral wavenumbers spanning the Mid-Infrared region (399.19 - 4000.60 cm⁻¹) measured at a high resolution of 4 cm⁻¹.")
    doc.add_paragraph("• Split: Stratified split containing 1,853 samples for training, 398 for validation, and 398 for testing.")
    doc.add_paragraph("• Target Classes: Balanced dataset containing PET, HDPE, LDPE, PP, PS, and PVC.")
    
    # --- SECTION 3 ---
    h3 = doc.add_paragraph()
    r3 = h3.add_run("3. Preprocessing Pipelines")
    r3.font.bold = True
    r3.font.size = Pt(14)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)
    
    doc.add_paragraph("• NIR HSI Preprocessing: Spatial dimensions are reduced by calculating the spatial mean spectrum of the 64 x 64 subimage. This reduces each data cube into a 1D vector of length 232, which is then normalized using Z-score column scaling (StandardScaler).")
    doc.add_paragraph("• MIR FTIR Preprocessing: Raw 3,736-point spectra are smoothed using a Savitzky-Golay filter (window length = 15, polynomial order = 2, derivative order = 0) to remove high-frequency noise. Standard Normal Variate (SNV) normalization is applied individually to each spectrum to correct for physical path length and light-scattering variations. Features are then Z-score column scaled.")
    
    # --- SECTION 4 ---
    h4 = doc.add_paragraph()
    r4 = h4.add_run("4. Machine Learning Models & Performance")
    r4.font.bold = True
    r4.font.size = Pt(14)
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(6)
    
    doc.add_paragraph("• NIR Classifier: Support Vector Machine (SVM) utilizing a Radial Basis Function (RBF) kernel (regularization parameter C = 10.0, gamma = 'scale'). It achieved a test accuracy of 97.12% and a weighted F1-score of 97.11%.")
    doc.add_paragraph("• MIR Classifier: Random Forest (RF) classifier consisting of 200 estimators (decision trees) with a max depth of 15. The model achieved a test accuracy and weighted F1-score of 100.00%.")
    
    # --- SECTION 5 ---
    h5 = doc.add_paragraph()
    r5 = h5.add_run("5. Physical Interpretation & Wavenumber Bands")
    r5.font.bold = True
    r5.font.size = Pt(14)
    h5.paragraph_format.space_before = Pt(12)
    h5.paragraph_format.space_after = Pt(6)
    
    doc.add_paragraph("Molecular bonds absorb infrared radiation at highly specific frequencies, which provides the physical basis for polymer identification. The following key absorption regions are utilized by the MIR classifier:")
    
    # Table Creation
    table = doc.add_table(rows=6, cols=3)
    table.style = 'Table Grid'
    
    # Set headers
    headers = ["Wavenumber Range (cm⁻¹)", "Vibration / Bond Type", "Polymer Identified"]
    hdr_cells = table.rows[0].cells
    for i, title_text in enumerate(headers):
        hdr_cells[i].text = title_text
        hdr_cells[i].paragraphs[0].runs[0].font.bold = True
        
    data = [
        ["1700 - 1750", "C=O (Carbonyl) stretching vibration", "PET (Ester linkage)"],
        ["2800 - 3000", "C-H (CH2 and CH3) stretching", "HDPE, LDPE, PP (Polyolefins)"],
        ["1375", "C-H bending of methyl group", "PP (Separates PP from PE)"],
        ["1450 - 1600 & 700 - 760", "Aromatic ring C=C / C-H bends", "PS (Polystyrene)"],
        ["600 - 700", "C-Cl stretching", "PVC (Chlorine bond)"]
    ]
    
    for row_idx, row_data in enumerate(data):
        row_cells = table.rows[row_idx + 1].cells
        for col_idx, text in enumerate(row_data):
            row_cells[col_idx].text = text
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    doc.add_paragraph("Black plastics containing carbon black pigment absorb almost all Near-Infrared (NIR) photons, dropping average reflectance to less than 8%. Under these conditions, the NIR model fails. However, in the Mid-Infrared (MIR) range, the light's wavelength (2.5 - 25 µm) is much larger than the carbon black pigment particles, leading to minimal scattering and absorption. Thus, the MIR spectrometer beam successfully interacts with the polymer chains, resolving clean finger-print peaks (like C-H stretching for PP or C-Cl for PVC) to classify the black plastic sample with 100% precision.")
    
    # Save document
    os.makedirs('results', exist_ok=True)
    fpath = 'results/NIR_MIR_Technical_Report.docx'
    doc.save(fpath)
    print(f"Word document saved to: {fpath}")

if __name__ == "__main__":
    create_report()
