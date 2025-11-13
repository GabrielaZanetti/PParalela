#!/usr/bin/env python3
"""Classifica letras de músicas usando um modelo local exposto pelo Ollama."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
API_PATH = "/api/generate"
SENTIMENT_MAP = {
    "positiva": "positive",
    "positive": "positive",
    "negativa": "negative",
    "negative": "negative",
    "neutra": "neutral",
    "neutral": "neutral",
}
PROMPT_TEMPLATE = (
    "Classifique o sentimento da letra abaixo em uma única palavra: Positiva, Neutra ou Negativa.\n"
    "Responda apenas com uma dessas palavras, sem explicações adicionais.\n"
    "Letra:\n"""\n{lyrics}\n"""\nFim da letra."
)


@dataclass
class SentimentTotals:
    positive: int = 0
    neutral: int = 0
    negative: int = 0

    def to_json(self) -> Dict[str, int]:
        return {
            "positive": self.positive,
            "neutral": self.neutral,
            "negative": self.negative,
        }

    def update(self, label: str) -> None:
        normalized = label.lower()
        if normalized not in {"positive", "neutral", "negative"}:
            LOGGER.warning("Rótulo de sentimento desconhecido '%s', contabilizando como neutro", label)
            normalized = "neutral"
        setattr(self, normalized, getattr(self, normalized) + 1)


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/") + API_PATH
        self.timeout = timeout

    def classify(self, model: str, lyrics: str, options: Optional[Dict[str, float]] = None) -> str:
        payload = {
            "model": model,
            "prompt": PROMPT_TEMPLATE.format(lyrics=lyrics.strip()),
            "stream": False,
        }
        if options:
            payload["options"] = options
        response = requests.post(self.base_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "").strip()
        normalized = raw.lower()
        for key, mapped in SENTIMENT_MAP.items():
            if key in normalized:
                return mapped
        LOGGER.warning("Resposta inesperada do modelo: '%s'", raw)
        return "neutral"


def iter_lyrics(path: str, limit: Optional[int] = None) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            lyric = line.strip()
            if not lyric:
                continue
            yield lyric


def classify_file(
    client: OllamaClient,
    model: str,
    lyrics_path: str,
    totals: SentimentTotals,
    retries: int,
    backoff: float,
    options: Optional[Dict[str, float]],
    limit: Optional[int],
) -> None:
    for lyric in iter_lyrics(lyrics_path, limit=limit):
        attempt = 0
        while True:
            try:
                label = client.classify(model, lyric, options)
                totals.update(label)
                break
            except requests.RequestException as exc:
                attempt += 1
                if attempt > retries:
                    LOGGER.error("Falha ao classificar letra após %d tentativas: %s", retries, exc)
                    totals.update("neutral")
                    break
                sleep_time = backoff * attempt
                LOGGER.warning(
                    "Erro na chamada ao Ollama (%s). Tentativa %d/%d. Aguardando %.1fs...",
                    exc,
                    attempt,
                    retries,
                    sleep_time,
                )
                time.sleep(sleep_time)


def parse_options(option_args: Optional[str]) -> Optional[Dict[str, float]]:
    if not option_args:
        return None
    options: Dict[str, float] = {}
    for chunk in option_args.split(","):
        if "=" not in chunk:
            LOGGER.warning("Ignorando opção inválida: %s", chunk)
            continue
        key, value = chunk.split("=", 1)
        try:
            options[key.strip()] = float(value)
        except ValueError:
            LOGGER.warning("Valor inválido para opção '%s': %s", key, value)
    return options or None


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classificação de sentimento de letras via Ollama")
    parser.add_argument("--model", required=True, help="Nome do modelo disponível no Ollama (ex.: llama3:8b)")
    parser.add_argument("--input", required=True, help="Arquivo de entrada com uma letra por linha")
    parser.add_argument("--output", required=True, help="Arquivo JSON de saída com totais agregados")
    parser.add_argument("--host", default=DEFAULT_HOST, help="URL base do servidor Ollama (default: %(default)s)")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Timeout em segundos por requisição (default: %(default)s)"
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Número de tentativas em caso de erro de rede (default: %(default)s)"
    )
    parser.add_argument(
        "--backoff", type=float, default=2.0, help="Multiplicador de backoff progressivo em segundos"
    )
    parser.add_argument(
        "--options",
        help="Opções extra para o modelo no formato chave=valor separadas por vírgula (ex.: num_ctx=4096,temperature=0.1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Processa apenas N letras (útil para testes rápidos)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de log",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    client = OllamaClient(base_url=args.host, timeout=args.timeout)
    sentiment_totals = SentimentTotals()
    options = parse_options(args.options)

    LOGGER.info("Iniciando classificação usando o modelo '%s' para o arquivo %s", args.model, args.input)
    classify_file(
        client=client,
        model=args.model,
        lyrics_path=args.input,
        totals=sentiment_totals,
        retries=args.retries,
        backoff=args.backoff,
        options=options,
        limit=args.limit,
    )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(sentiment_totals.to_json(), handle, ensure_ascii=False, indent=2)
    LOGGER.info(
        "Classificação concluída: %s",
        json.dumps(sentiment_totals.to_json(), ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
