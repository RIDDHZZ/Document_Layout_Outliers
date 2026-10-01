# Block A – Dataset, Extraction, Features

## Run
```bash
pip install -r requirements.txt          # also needs the Tesseract binary for the OCR fallback
python -m ml.generate_dataset            # 120 originals -> 360 PDFs + data/metadata/annotations.csv
python -m ml.build_features              # -> data/features/page_features.csv + descriptive sanity report
python -m pytest tests -q                # 10 tests
```
Everything is seeded from `config.yaml` (`random_state: 42`); two runs give identical annotations (tested).

## Files
| file | role |
|---|---|
| `ml/generate_dataset.py` | 5 legal document types (contract, affidavit, notice, application form, certificate); margin-altered and signature-inserted variants; ground truth |
| `ml/extract.py` | PDF -> `PageLayout` (normalised boxes). Native text/image blocks; Tesseract + OpenCV fallback for scanned pages |
| `ml/features.py` | 83 page features in 5 groups + per-element table for region localisation |
| `ml/build_features.py` | runs the pipeline over the dataset, prints feature-shift report |

## Dataset produced
120 originals -> 214 normal pages; 174 anomalous margin pages (+39 unaltered pages inside margin variants);
120 anomalous signature pages (+94 unaltered). `annotations.csv` has one row per (file, page) with
`document_id, source_doc_id, category, page_number, page_width/height, anomaly_type, x1..y2 (PDF points, top-left origin), is_anomalous, alteration_detail`.
`source_doc_id` ties an original to its variants -> **split by it** in training to avoid leakage.

## Features (83)
* **margin (26):** per side (left/right/top/bottom) mean, median, std, var, min, max of element edges, plus `margin_ratio_lr`, `margin_ratio_tb`. Text elements only.
* **bbox (17):** width/height/area mean, std, variance; aspect mean/std; centre-x/y mean, std, variance.
* **density (22):** 4x4 normalised cell densities; mean raw count per cell, std, var, max, min, entropy.
* **counts (2):** `n_elements`, `n_text_elements`.
* **signature (16):** min/max aggregates over non-text elements: count, area, position, gap to text, distance to text-block bottom/right, overlap with text, local density.

## Findings so far (descriptive; no model yet)
* Margin alterations are measurable on reasonably full pages (>=15 text lines): 94-100% of pages lie >2 sd from normal on the matching feature, for every mode except `bottom_change`.
* Inserted signatures separate from genuine ones on gap/overlap in 85-100% of pages, depending on placement.

## Known limitations (state these in the report)
1. **Right/bottom margins are unobservable on sparse pages** (no full-width line / page not full). On all pages, right-margin modes show only 0-9% separation vs 94-100% on dense pages. Expect lower recall there.
2. **`bottom_change` is almost absent** (3 pages): the generator keeps a variant only if the page measurably changed, and bottom margin rarely does. Report it as excluded/rare.
3. **OCR path is a fallback, not equivalent to native.** Text-line counts and horizontal features agree well (margins ~0.03 sd), but Tesseract returns tight ink boxes vs PyMuPDF font boxes, so `bbox_height_*` differs ~4 sd. For scanned uploads either train an OCR-domain model (`build_features --force-ocr` on the training set) or drop height features. OCR-path signature detection found 7/12 signatures in a spot check (misses when the signature overlaps body text and is masked with it).
4. Native extraction reads the PDF's text layer and image blocks. Hidden/invisible text is not detected.
5. Genuine and inserted signatures are the same procedural scribble generator: the detector learns *placement*, not handwriting.
