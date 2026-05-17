"""
Script de evaluación RAGAS para el pipeline Pymerag.

Calcula las métricas estándar de RAGAS (faithfulness, answer_relevancy,
context_precision, context_recall) sobre un golden set de QA pairs
y escribe los resultados en eval/reports/.

Uso:
    python eval/run_ragas.py                          # Evaluación offline
    python eval/run_ragas.py --live                   # Evaluación con pipeline real
    python eval/run_ragas.py --output reports/mi_eval.json  # Ruta de salida personalizada
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ragas_eval")

# ── Constantes ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "golden_set.jsonl"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
DEFAULT_OUTPUT = REPORTS_DIR / "ragas_results.json"


# ── Carga del golden set ──────────────────────────────────────────────


def load_golden_set(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Carga el golden set desde un archivo JSONL.

    Args:
        path: Ruta al archivo JSONL. Si es None, usa el default.

    Returns:
        Lista de diccionarios con preguntas, ground truth y contextos.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o mal formado.
    """
    file_path = path or GOLDEN_SET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Golden set no encontrado: {file_path}")

    items: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Error en línea {line_num} de {file_path}: {exc}"
                ) from exc

    if not items:
        raise ValueError(f"Golden set vacío: {file_path}")

    logger.info("Cargados %d pares QA del golden set '%s'.", len(items), file_path.name)
    return items


# ── Evaluación offline (sin pipeline real) ────────────────────────────


def evaluate_offline(golden_set: list[dict[str, Any]]) -> dict[str, Any]:
    """Evalúa el golden set usando respuestas pre-calculadas del dataset.

    En este modo, las respuestas del ground_truth_answer se usan como
    las respuestas "generadas" por el sistema. Las métricas RAGAS se
    simulan con heurísticas basadas en solapamiento de texto.

    Args:
        golden_set: Lista de QA pairs con ground truth.

    Returns:
        Diccionario con métricas agregadas y resultados por ítem.
    """
    per_item: list[dict[str, Any]] = []
    total_items = len(golden_set)

    for item in golden_set:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]
        contexts = item.get("ground_truth_contexts", [])

        # En modo offline, usamos el ground truth como "respuesta generada"
        answer = ground_truth

        # ── Cálculo de métricas heurísticas ────────────────────────

        # Faithfulness: proporción de afirmaciones en la respuesta que
        # están soportadas por los contextos. Heurística: solapamiento de tokens.
        faithfulness = _compute_token_overlap(answer, " ".join(contexts))

        # Answer Relevancy: qué tan relevante es la respuesta para la pregunta.
        # Heurística: solapamiento de tokens entre pregunta y respuesta.
        answer_relevancy = _compute_token_overlap(answer, question)

        # Context Precision: proporción de chunks recuperados que son relevantes.
        # Heurística: solapamiento promedio pregunta-contexto.
        context_precision = (
            sum(_compute_token_overlap(question, ctx) for ctx in contexts) / len(contexts)
            if contexts
            else 0.0
        )

        # Context Recall: proporción del ground truth cubierta por los contextos.
        # Heurística: solapamiento entre ground truth y contextos.
        context_recall = _compute_token_overlap(
            ground_truth, " ".join(contexts)
        )

        per_item.append(
            {
                "id": item.get("id", ""),
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "faithfulness": round(faithfulness, 4),
                "answer_relevancy": round(answer_relevancy, 4),
                "context_precision": round(context_precision, 4),
                "context_recall": round(context_recall, 4),
                "category": item.get("category", ""),
            }
        )

    # ── Agregación ─────────────────────────────────────────────────
    aggregated = {
        "faithfulness_avg": round(
            sum(r["faithfulness"] for r in per_item) / total_items, 4
        ),
        "answer_relevancy_avg": round(
            sum(r["answer_relevancy"] for r in per_item) / total_items, 4
        ),
        "context_precision_avg": round(
            sum(r["context_precision"] for r in per_item) / total_items, 4
        ),
        "context_recall_avg": round(
            sum(r["context_recall"] for r in per_item) / total_items, 4
        ),
        "total_items": total_items,
    }

    return {
        "metadata": {
            "evaluation_mode": "offline",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": str(GOLDEN_SET_PATH.name),
            "total_items": total_items,
        },
        "aggregated_metrics": aggregated,
        "per_item_results": per_item,
    }


def _compute_token_overlap(text_a: str, text_b: str) -> float:
    """Calcula el coeficiente de solapamiento de tokens entre dos textos.

    Args:
        text_a: Primer texto.
        text_b: Segundo texto.

    Returns:
        Valor entre 0.0 y 1.0 indicando la proporción de tokens
        de text_a que aparecen en text_b.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a:
        return 0.0

    overlap = len(tokens_a & tokens_b)
    return overlap / len(tokens_a)


# ── Evaluación con pipeline real ──────────────────────────────────────


def evaluate_live(golden_set: list[dict[str, Any]]) -> dict[str, Any]:
    """Evalúa el golden set usando el pipeline RAG real de Pymerag.

    Requiere que el servidor FastAPI esté corriendo y los servicios
    externos (Qdrant, LLM) estén disponibles.

    Args:
        golden_set: Lista de QA pairs.

    Returns:
        Diccionario con métricas agregadas y resultados por ítem.
    """
    try:
        import httpx
    except ImportError:
        logger.error(
            "httpx no está instalado. Instálalo con: pip install httpx"
        )
        sys.exit(1)

    from app.core.config import settings

    API_URL = f"http://{settings.api_host}:{settings.api_port}/api/v1/query"

    per_item: list[dict[str, Any]] = []
    total_items = len(golden_set)

    # Intentar cargar ragas para métricas reales
    ragas_available = False
    faithfulness = None  # type: ignore[assignment]
    answer_relevancy = None  # type: ignore[assignment]
    context_precision = None  # type: ignore[assignment]
    context_recall = None  # type: ignore[assignment]
    evaluator_llm = None
    try:
        from ragas.metrics import (  # type: ignore[no-redef]
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
        from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="deepseek-v4",
                openai_api_base=settings.llm_api_base,
                openai_api_key="not-needed",
            )
        )
        ragas_available = True
        logger.info("RAGAS disponible — usando métricas reales con LLM evaluador.")
    except ImportError:
        logger.warning(
            "RAGAS no instalado. Usando métricas heurísticas. "
            "Instala con: pip install ragas langchain-openai"
        )

    with httpx.Client(timeout=120.0) as client:
        for item in golden_set:
            question = item["question"]
            ground_truth = item["ground_truth_answer"]
            contexts_gt = item.get("ground_truth_contexts", [])

            item_result: dict[str, Any] = {
                "id": item.get("id", ""),
                "question": question,
                "ground_truth": ground_truth,
                "category": item.get("category", ""),
            }

            try:
                resp = client.post(
                    API_URL,
                    json={
                        "query": question,
                        "top_k": 5,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "")
                sources = data.get("sources", [])
                contexts = [s.get("content", "") for s in sources]

                item_result["answer"] = answer
                item_result["retrieved_contexts"] = contexts

                if ragas_available:
                    # RAGAS real
                    from datasets import Dataset

                    ds = Dataset.from_dict(
                        {
                            "question": [question],
                            "answer": [answer],
                            "contexts": [contexts],
                            "ground_truth": [ground_truth],
                        }
                    )

                    try:
                        faith = faithfulness.score(ds, evaluator_llm)[0]
                        relev = answer_relevancy.score(ds, evaluator_llm)[0]
                        cprec = context_precision.score(ds, evaluator_llm)[0]
                        crecl = context_recall.score(ds, evaluator_llm)[0]
                    except Exception as exc:
                        logger.warning("RAGAS falló en ítem %s: %s", item["id"], exc)
                        faith = 0.0
                        relev = 0.0
                        cprec = 0.0
                        crecl = 0.0
                else:
                    # Heurísticas
                    faith = _compute_token_overlap(
                        answer, " ".join(contexts)
                    )
                    relev = _compute_token_overlap(answer, question)
                    cprec = (
                        sum(
                            _compute_token_overlap(question, ctx)
                            for ctx in contexts
                        )
                        / len(contexts)
                        if contexts
                        else 0.0
                    )
                    crecl = _compute_token_overlap(
                        ground_truth, " ".join(contexts)
                    )

                item_result["faithfulness"] = round(faith, 4)
                item_result["answer_relevancy"] = round(relev, 4)
                item_result["context_precision"] = round(cprec, 4)
                item_result["context_recall"] = round(crecl, 4)

            except httpx.HTTPError as exc:
                logger.error("Error consultando API para '%s': %s", item["id"], exc)
                item_result["error"] = str(exc)
                item_result["faithfulness"] = 0.0
                item_result["answer_relevancy"] = 0.0
                item_result["context_precision"] = 0.0
                item_result["context_recall"] = 0.0

            per_item.append(item_result)

    # ── Agregación ─────────────────────────────────────────────────
    valid_items = [r for r in per_item if "error" not in r]
    n = len(valid_items) or 1

    aggregated = {
        "faithfulness_avg": round(
            sum(r.get("faithfulness", 0.0) for r in valid_items) / n, 4
        ),
        "answer_relevancy_avg": round(
            sum(r.get("answer_relevancy", 0.0) for r in valid_items) / n, 4
        ),
        "context_precision_avg": round(
            sum(r.get("context_precision", 0.0) for r in valid_items) / n, 4
        ),
        "context_recall_avg": round(
            sum(r.get("context_recall", 0.0) for r in valid_items) / n, 4
        ),
        "total_items": total_items,
        "valid_items": len(valid_items),
        "errors": total_items - len(valid_items),
    }

    return {
        "metadata": {
            "evaluation_mode": "live",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": str(GOLDEN_SET_PATH.name),
            "total_items": total_items,
            "api_url": API_URL,
            "ragas_available": ragas_available,
        },
        "aggregated_metrics": aggregated,
        "per_item_results": per_item,
    }


# ── Persistencia ──────────────────────────────────────────────────────


def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Guarda los resultados de evaluación en formato JSON.

    Args:
        results: Diccionario con métricas y resultados.
        output_path: Ruta de destino del archivo JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Resultados guardados en: %s", output_path)


def print_summary(results: dict[str, Any]) -> None:
    """Imprime un resumen de los resultados en consola.

    Args:
        results: Diccionario con métricas y resultados.
    """
    meta = results["metadata"]
    agg = results["aggregated_metrics"]

    print("\n" + "=" * 60)
    print("  RESULTADOS DE EVALUACIÓN RAGAS — Pymerag")
    print("=" * 60)
    print(f"  Modo:          {meta['evaluation_mode']}")
    print(f"  Dataset:       {meta['dataset']}")
    print(f"  Ítems totales: {meta['total_items']}")
    print(f"  Timestamp:     {meta['timestamp']}")
    print("-" * 60)
    print("  MÉTRICAS AGREGADAS:")
    print(f"    Faithfulness:        {agg['faithfulness_avg']:.4f}")
    print(f"    Answer Relevancy:    {agg['answer_relevancy_avg']:.4f}")
    print(f"    Context Precision:   {agg['context_precision_avg']:.4f}")
    print(f"    Context Recall:      {agg['context_recall_avg']:.4f}")
    print("-" * 60)

    # Mostrar detalle por ítem
    for item in results.get("per_item_results", []):
        status = "✓" if "error" not in item else "✗"
        print(
            f"  {status} [{item.get('id', '?')}] "
            f"F:{item.get('faithfulness', 0):.3f} "
            f"AR:{item.get('answer_relevancy', 0):.3f} "
            f"CP:{item.get('context_precision', 0):.3f} "
            f"CR:{item.get('context_recall', 0):.3f} "
            f"— {item.get('question', '')[:60]}..."
        )

    # Errores
    errors = [i for i in results.get("per_item_results", []) if "error" in i]
    if errors:
        print("-" * 60)
        print("  ERRORES:")
        for e in errors:
            print(f"    [{e.get('id', '?')}] {e['error']}")

    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evaluación RAGAS del pipeline Pymerag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python eval/run_ragas.py                          # Modo offline\n"
            "  python eval/run_ragas.py --live                   # Modo live\n"
            "  python eval/run_ragas.py --output mi_eval.json    # Salida personalizada\n"
            "  python eval/run_ragas.py --dataset custom.jsonl   # Golden set personalizado\n"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Ejecutar evaluación contra el pipeline RAG real (requiere servidor).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del archivo JSON de salida. (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"Ruta al archivo JSONL del golden set. (default: {GOLDEN_SET_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprimir salida de consola (solo guarda archivo).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal del script de evaluación."""
    args = parse_args()

    # 1. Cargar golden set
    golden_set = load_golden_set(args.dataset)

    # 2. Ejecutar evaluación
    mode = "live" if args.live else "offline"
    logger.info("Iniciando evaluación RAGAS en modo '%s'...", mode)

    if args.live:
        results = evaluate_live(golden_set)
    else:
        results = evaluate_offline(golden_set)

    # 3. Guardar resultados
    save_results(results, args.output)

    # 4. Mostrar resumen
    if not args.quiet:
        print_summary(results)


if __name__ == "__main__":
    main()
