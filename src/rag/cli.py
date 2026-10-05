"""Buyruq satri interfeysi: python -m rag {parse,build,query}"""
import argparse
import json
import sys

from . import config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag", description="Lokal offline RAG tizimi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help="PDF dan matn ajratib olish")
    p.add_argument("pdf")
    p.add_argument("--out", default=str(config.PAGES_JSON))
    p.add_argument("--md", default=None, help="Markdown nusxasini ham saqlash")

    b = sub.add_parser("build", help="Qidiruv indeksini qurish")
    b.add_argument("--pages", default=str(config.PAGES_JSON))
    b.add_argument("--index-dir", default=str(config.INDEX_DIR))
    b.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    b.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)

    q = sub.add_parser("query", help="Savol bo'yicha eng mos bo'laklarni topish")
    q.add_argument("text", nargs="+")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--index-dir", default=str(config.INDEX_DIR))
    q.add_argument("--prompt", action="store_true", help="LLM uchun tayyor promptni chiqarish")
    q.add_argument("--json", action="store_true", help="Natijani JSON ko'rinishida chiqarish")

    args = ap.parse_args(argv)

    if args.cmd == "parse":
        from .parse_pdf import parse_pdf
        parse_pdf(args.pdf, args.out, args.md)
    elif args.cmd == "build":
        from .indexer import build_index
        build_index(args.pages, args.index_dir, args.chunk_size, args.overlap)
    else:
        from .retriever import Retriever
        res = Retriever(args.index_dir).answer(" ".join(args.text), k=args.k)
        if args.json:
            json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
            print()
        elif args.prompt:
            print(res["prompt"])
        else:
            print("SAVOL:", res["query"], "\n")
            for r in res["retrieved"]:
                print(f"[Sahifa {r['page']} | skor={r['score']:.3f}]\n{r['text']}\n{'-' * 70}")


if __name__ == "__main__":
    main()
