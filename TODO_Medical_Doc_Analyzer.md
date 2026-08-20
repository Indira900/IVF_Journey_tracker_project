# AI Medical Document Analyzer — Implementation TODO

## Step 1: Create `medical_document_analyzer.py` (NEW)
- [ ] File handling with fallbacks (PDF → OCR, DOCX → docx2txt, Images → PIL + pytesseract)
- [ ] Text cleaning and normalization
- [ ] Comprehensive regex extraction for all 12 medical parameters
- [ ] BMI calculation from height/weight
- [ ] Medical reference range analysis + abnormality detection
- [ ] Summary generation (short, detailed, issues, recommendations)
- [ ] Model input JSON mapping
- [ ] Error handling (no hallucination, "No medical data detected")

## Step 2: Update `main.py`
- [ ] Import and integrate medical_document_analyzer
- [ ] Replace old extract_medical_params() and generate_medical_insights()
- [ ] Update upload_document route with new analyzer
- [ ] Update analyze_document route with structured analysis
- [ ] Add /api/analyze_document/<int:doc_id> API route
- [ ] Add /auto_fill_profile route for one-click profile update

## Step 3: Update `templates/document_analysis.html`
- [ ] Extracted Data JSON display with normal/abnormal badges
- [ ] Analysis section with reference range comparisons
- [ ] Summary section (short, detailed, issues, recommendations)
- [ ] Model Input JSON display
- [ ] Auto-fill Profile button
- [ ] Collapsible original extracted text

## Step 4: Update `templates/my_documents.html`
- [ ] Enhanced AI-Extracted Medical Summary panel
- [ ] Analysis badges per document
- [ ] Quick-action buttons

## Step 5: Update `requirements.txt`
- [ ] Add PyMuPDF, docx2txt dependencies

## Step 6: Testing & Verification
- [ ] Install new dependencies
- [ ] Test with existing uploaded documents
- [ ] Verify Flask app runs correctly

