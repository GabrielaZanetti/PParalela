#!/usr/bin/env python3
"""Converte o dataset original do Spotify para o formato TSV usado pelo aplicativo MPI."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
PREFERRED_TRACK_KEYS: Sequence[str] = ("track", "track_name", "song_name")
PREFERRED_ARTIST_KEYS: Sequence[str] = ("artist", "artist_name")
PREFERRED_LYRICS_KEYS: Sequence[str] = ("lyrics", "lyric", "text")


def find_first(row: Mapping[str, str], candidates: Sequence[str]) -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"Nenhuma chave encontrada entre {candidates}")


def sanitize(text: str) -> str:
    return (
        text.replace("\t", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def preprocess(input_path: str, output_path: str, limit: int | None = None) -> int:
    total_rows = 0
    written_rows = 0
    with open(input_path, "r", encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = target
        for row in reader:
            total_rows += 1
            if limit is not None and written_rows >= limit:
                break
            try:
                track = sanitize(find_first(row, PREFERRED_TRACK_KEYS))
                artist = sanitize(find_first(row, PREFERRED_ARTIST_KEYS))
                lyrics = sanitize(find_first(row, PREFERRED_LYRICS_KEYS))
            except KeyError as exc:
                LOGGER.debug("Linha %d ignorada (%s)", total_rows, exc)
                continue
            if not lyrics:
                continue
            writer.write(f"{track}\t{artist}\t{lyrics}\n")
            written_rows += 1
    LOGGER.info("Processadas %d linhas, escritas %d", total_rows, written_rows)
    return written_rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera um TSV simples (track\tartist\tlyrics) a partir do CSV original do Spotify"
    )
    parser.add_argument("input", help="Arquivo CSV original baixado do Kaggle")
    parser.add_argument(
        "output",
        nargs="?",
        default="spotify_lyrics.tsv",
        help="Arquivo TSV de saída (default: %(default)s)",
    )
    parser.add_argument("--limit", type=int, help="Processa apenas as N primeiras linhas (teste)")
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not os.path.exists(args.input):
        LOGGER.error("Arquivo de entrada %s não encontrado", args.input)
        return 1

    preprocess(args.input, args.output, limit=args.limit)
    LOGGER.info("Arquivo TSV gerado em %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
