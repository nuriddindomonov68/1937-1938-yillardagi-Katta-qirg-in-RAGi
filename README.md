# Katta qirg'in RAG

[![CI](https://github.com/<foydalanuvchi>/katta-qirgin-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/<foydalanuvchi>/katta-qirgin-rag/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

O'zbek tilidagi (lotin yozuvi) PDF hujjatlar uchun **to'liq lokal va offline** RAG (Retrieval-Augmented Generation) tizimi. API kalit, internet yoki GPU kerak emas.

Namuna sifatida *"1937–1938-yillardagi «Katta qirg'in» tarixining muhim sanalari"* ma'lumotnomasi (40 sahifa) bilan keladi, lekin istalgan PDF bilan ishlaydi.

## Imkoniyatlar

- **PDF → matn**: `docling-parse` orqali sahifa-sahifa ajratib olish
- **Chunking**: so'z chegarasiga moslangan, qoplamali (overlap) bo'laklar
- **Qidiruv**: TF-IDF (1–2 so'zli n-gram) + kosinus o'xshashlik
- **O'zbekchaga moslashgan**: `o'`, `g'` kabi apostroflar (`’`, `‘`, `ʻ`, `` ` ``) birxillashtiriladi va so'z ichida saqlanadi
- **Saqlash**: bo'laklar SQLite da, vektorlashtirgich diskda; qayta ishga tushirishda qayta qurish shart emas
- **LLM tayyor prompt**: topilgan kontekst + sahifa raqamlari bilan, istalgan LLM ga yuborishga tayyor
- Buyruq satri (CLI), Python API va testlar (pytest + GitHub Actions)

## O'rnatish

```bash
git clone https://github.com/<foydalanuvchi>/katta-qirgin-rag.git
cd katta-qirgin-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .                  # asosiy
pip install -e ".[parse]"         # PDF parse qilish kerak bo'lsa
```

## Tezkor boshlash

Repoda tayyor `data/parsed_pages.json` bor, shuning uchun to'g'ridan-to'g'ri:

```bash
rag build                                              # indeks qurish (index/ papkasiga)
rag query "1937-yil 30-iyuldagi 00447-sonli buyruq nima haqida?"
rag query "00447-sonli buyruq" -k 3 --prompt           # LLM uchun tayyor prompt
rag query "00447-sonli buyruq" --json                  # JSON chiqish
```

(`rag` buyrug'i ishlamasa: `python -m rag ...`, o'rnatishsiz esa `PYTHONPATH=src python -m rag ...`.)

### O'z PDF ingiz bilan

```bash
rag parse hujjat.pdf --md data/parsed.md     # -> data/parsed_pages.json
rag build
rag query "savolingiz"
```

## Python API

```python
from rag import Retriever

r = Retriever()                       # indeksni bir marta yuklaydi
res = r.answer("00447-sonli buyruq nima haqida?", k=5)

for hit in res["retrieved"]:
    print(hit["page"], hit["score"], hit["text"][:100])

prompt = res["prompt"]                # istalgan LLM ga yuboring
```

## Sozlamalar

Muhit o'zgaruvchilari (yoki `rag build --chunk-size 500 --overlap 100`):

| O'zgaruvchi | Standart | Izoh |
|---|---|---|
| `RAG_PAGES_JSON` | `data/parsed_pages.json` | Kiruvchi sahifalar fayli |
| `RAG_INDEX_DIR` | `index/` | Indeks papkasi |
| `RAG_CHUNK_SIZE` | `700` | Bo'lak uzunligi (belgi) |
| `RAG_CHUNK_OVERLAP` | `150` | Bo'laklar orasidagi qoplama |

## Loyiha tuzilishi

```
src/rag/
  parse_pdf.py   PDF -> parsed_pages.json
  text_utils.py  normalizatsiya va chunking
  indexer.py     indeks qurish (SQLite + TF-IDF)
  retriever.py   qidiruv va prompt tayyorlash
  cli.py         buyruq satri
data/            namunaviy kiruvchi ma'lumot
tests/           pytest testlar
```

## Cheklovlar

- TF-IDF **leksik** qidiruv: sinonimlar va ma'no jihatidan yaqin so'zlarni tushunmaydi. Semantik natija kerak bo'lsa, `sentence-transformers` kabi embedding modeliga o'tish mumkin (`Retriever` ni almashtirish kifoya).
- PDF dan olingan matnda OCR xatolari bor (masalan, `bo'Igan`, `togrisida`). Yaxshiroq OCR natijaning sifatini oshiradi.
- `docling-parse` faqat matnli qatlamni o'qiydi; skaner qilingan PDF lar avval OCR qilinishi kerak.
- Indeks `pickle` da saqlanadi — faqat o'zingiz qurgan fayllarni yuklang, begona `index/` fayllarini ochmang.

## Testlar

```bash
pip install -e ".[dev]"
pytest
```

## Litsenziya va ma'lumot huquqlari

Kod — [MIT](LICENSE). `data/parsed_pages.json` dagi matn manba nashrga (Toshkent, "Fan" nashriyoti, 2013) tegishli va faqat demo uchun berilgan; ommaga ochishdan oldin huquqiy holatini tekshiring yoki faylni o'chirib, o'z ma'lumotingizni qo'ying.
