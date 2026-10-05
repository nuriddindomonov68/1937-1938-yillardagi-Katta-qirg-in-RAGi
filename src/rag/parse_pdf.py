"""PDF dan matnni docling-parse yordamida sahifa-sahifa ajratib olish.

Eslatma: bu to'liq Docling pipeline (layout/OCR modellari) emas, balki uning
past darajadagi PDF backend'i (docling-parse). Skanerlangan PDF lar uchun
avval OCR qilingan bo'lishi kerak.
"""
import json
from pathlib import Path


def parse_pdf(src: Path, out_json: Path, out_md: Path | None = None) -> int:
    try:
        from docling_parse.pdf_parser import DoclingPdfParser
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "docling-parse o'rnatilmagan. O'rnating: pip install -r requirements-parse.txt"
        ) from e

    src, out_json = Path(src), Path(out_json)
    if not src.exists():
        raise FileNotFoundError(src)

    pdf_doc = DoclingPdfParser().load(path_or_stream=str(src))
    n_pages = pdf_doc.number_of_pages()
    pages = []
    for page_no in range(1, n_pages + 1):
        page = pdf_doc.get_page(page_no)
        lines = [tl.text.strip() for tl in page.textline_cells if tl.text.strip()]
        pages.append({"page": page_no, "text": "\n".join(lines)})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_md:
        out_md = Path(out_md)
        out_md.write_text(
            "".join(f"\n\n## Sahifa {p['page']}\n\n{p['text']}" for p in pages), encoding="utf-8"
        )
    print(f"Sahifalar: {n_pages}, belgilar: {sum(len(p['text']) for p in pages)}")
    return n_pages
