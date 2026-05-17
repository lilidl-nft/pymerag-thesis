"""
Entrenador de tópicos BERTopic para el corpus de Pymerag.

Carga chunks desde Qdrant, entrena un modelo BERTopic (o un fallback
simple basado en frecuencia de términos si BERTopic no está instalado)
y persiste los tópicos descubiertos en PostgreSQL.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from sqlmodel import Session, create_engine, delete

from app.core.config import settings
from app.models.sql import Topic
from app.rag.retriever import QdrantRetriever

logger = logging.getLogger(__name__)

# ── Constantes ──────────────────────────────────────────────────────

MIN_CHUNKS_FOR_TRAINING = 10
"""Número mínimo de chunks requeridos para entrenar el modelo de tópicos."""

SCROLL_BATCH_SIZE = 1000
"""Tamaño de lote para la paginación de scroll en Qdrant."""

MAX_REPRESENTATIVE_CHUNKS = 10
"""Número máximo de chunks representativos por tópico."""

_STOP_WORDS_ES: set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "por", "para", "como", "que",
    "es", "son", "fue", "eran", "ha", "han", "había",
    "su", "sus", "al", "lo", "le", "se", "y", "e", "o", "u",
    "ni", "pero", "aunque", "porque", "si", "no",
    "más", "menos", "muy", "tan", "todo", "toda",
    "entre", "hasta", "desde", "sobre", "bajo", "ante", "tras",
    "durante", "mediante",
}
"""Stop words en español para el fallback simple."""

_STOP_WORDS_EN: set[str] = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were",
    "been", "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "shall", "it", "its", "he", "she", "they", "we", "you",
    "i", "me", "him", "her", "us", "them", "his", "her",
    "their", "our", "my", "your", "and", "or", "but", "not",
    "this", "that", "these", "those", "there", "here",
    "which", "who", "whom", "what", "when", "where", "why",
    "how", "all", "each", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "only", "own",
    "same", "so", "than", "too", "very", "just", "about",
    "into", "through", "during", "before", "after",
    "above", "below", "up", "down",
}
"""Stop words en inglés para el fallback simple."""

_ALL_STOP_WORDS: set[str] = _STOP_WORDS_ES | _STOP_WORDS_EN


# ── Clase principal ─────────────────────────────────────────────────


class TopicTrainer:
    """Entrena y persiste tópicos del corpus usando BERTopic o fallback.

    Responsabilidades:
    - Cargar todos los chunks indexados desde Qdrant.
    - Entrenar BERTopic (o clustering léxico simple como fallback).
    - Persistir los tópicos descubiertos en PostgreSQL.
    - Retornar un resumen estructurado de los tópicos generados.
    """

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
    ) -> None:
        """Inicializa el entrenador de tópicos.

        Args:
            retriever: Recuperador de Qdrant (crea uno nuevo si es None).
        """
        self._retriever = retriever or QdrantRetriever()
        self._engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )

    # ── API pública ──────────────────────────────────────────────────

    def train(self) -> list[dict[str, Any]]:
        """Ejecuta el pipeline completo de descubrimiento de tópicos.

        Flujo:
        1. Cargar chunks desde Qdrant vía scroll paginado.
        2. Entrenar modelo (BERTopic o fallback si no está instalado).
        3. Limpiar tópicos anteriores de la base de datos.
        4. Persistir los nuevos tópicos descubiertos.
        5. Retornar lista de resúmenes.

        Returns:
            Lista de diccionarios con 'id', 'name', 'description',
            'chunk_count' y 'representative_chunks'.
        """
        # ── 1. Cargar chunks ──────────────────────────────────────
        chunk_texts, chunk_ids = self._load_all_chunks()
        if len(chunk_texts) < MIN_CHUNKS_FOR_TRAINING:
            logger.warning(
                "Chunks insuficientes para entrenar (%d < %d). "
                "Se requieren al menos %d chunks indexados.",
                len(chunk_texts),
                MIN_CHUNKS_FOR_TRAINING,
                MIN_CHUNKS_FOR_TRAINING,
            )
            return []

        logger.info(
            "Iniciando entrenamiento de tópicos con %d chunks.",
            len(chunk_texts),
        )

        # ── 2. Entrenar modelo ────────────────────────────────────
        try:
            topic_ids, topic_info = self._fit_bertopic(chunk_texts)
        except ImportError:
            logger.info(
                "BERTopic no está instalado; activando fallback simple."
            )
            topic_ids, topic_info = self._fit_fallback(chunk_texts)

        if not topic_info:
            logger.warning("No se descubrieron tópicos.")
            return []

        # ── 3-4. Agrupar y persistir ──────────────────────────────
        summaries = self._build_and_persist(
            topic_ids, topic_info, chunk_ids
        )

        logger.info(
            "Entrenamiento completado: %d tópicos descubiertos.",
            len(summaries),
        )
        return summaries

    # ── Carga de chunks ──────────────────────────────────────────────

    def _load_all_chunks(self) -> tuple[list[str], list[str]]:
        """Carga todos los chunks desde la colección Qdrant vía scroll.

        Returns:
            Tupla (textos, ids) con las listas de contenidos textuales
            y los IDs de los puntos en Qdrant.
        """
        texts: list[str] = []
        ids: list[str] = []

        try:
            offset: str | int | None = None
            while True:
                records, next_offset = self._retriever.client.scroll(
                    collection_name=self._retriever.collection_name,
                    scroll_filter=None,
                    limit=SCROLL_BATCH_SIZE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not records:
                    break

                for record in records:
                    payload = record.payload or {}
                    content = payload.get("content", "")
                    if content:
                        texts.append(content)
                        ids.append(str(record.id))

                if next_offset is None:
                    break
                offset = next_offset

        except Exception as exc:
            logger.error("Error al cargar chunks desde Qdrant: %s", exc)
            return [], []

        logger.info("Cargados %d chunks desde Qdrant.", len(texts))
        return texts, ids

    # ── Entrenamiento con BERTopic ───────────────────────────────────

    def _fit_bertopic(
        self, texts: list[str]
    ) -> tuple[list[int], dict[int, dict[str, Any]]]:
        """Entrena BERTopic sobre los textos provistos.

        Args:
            texts: Lista de contenidos de chunks.

        Returns:
            Tupla (asignaciones, info_tópicos):
            - asignaciones: ID de tópico (int) para cada chunk.
            - info_tópicos: diccionario {topic_id: {name, description, count}}.

        Raises:
            ImportError: Si BERTopic no está instalado.
        """
        try:
            from bertopic import BERTopic  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "BERTopic no está instalado. Instálalo con: "
                "pip install bertopic scikit-learn"
            )

        logger.info(
            "Inicializando BERTopic (modelo de embeddings: %s)...",
            settings.embedding_model,
        )

        topic_model = BERTopic(
            embedding_model=settings.embedding_model,
            min_topic_size=5,
            nr_topics="auto",
            verbose=True,
        )

        topics, _probs = topic_model.fit_transform(texts)

        # Extraer información de cada tópico desde el dataframe de BERTopic
        info: dict[int, dict[str, Any]] = {}
        try:
            topic_info_df = topic_model.get_topic_info()

            for _, row in topic_info_df.iterrows():
                topic_id = int(row["Topic"])
                if topic_id == -1:
                    continue  # ignorar outliers

                rep_words = row.get("Representation", [])
                if isinstance(rep_words, list) and rep_words:
                    name = ", ".join(rep_words[:5])
                else:
                    name = f"topic_{topic_id}"

                description = str(row.get("Name", name))
                count = int(row.get("Count", 0))

                info[topic_id] = {
                    "name": name,
                    "description": description,
                    "count": count,
                }

        except Exception as exc:
            logger.warning(
                "Error extrayendo info de tópicos: %s. Usando nombres genéricos.",
                exc,
            )
            # Fallback: construir info manual desde las asignaciones
            from collections import Counter as _Counter

            topic_counter = _Counter(topics)
            for topic_id, count in topic_counter.items():
                if topic_id == -1:
                    continue
                info[topic_id] = {
                    "name": f"topic_{topic_id}",
                    "description": f"Tópico automático {topic_id}",
                    "count": count,
                }

        logger.info(
            "BERTopic completado: %d tópicos (%d outliers).",
            len(info),
            sum(1 for t in topics if t == -1),
        )
        return topics, info

    # ── Fallback simple (sin BERTopic) ───────────────────────────────

    def _fit_fallback(
        self, texts: list[str]
    ) -> tuple[list[int], dict[int, dict[str, Any]]]:
        """Agrupa chunks por palabras clave frecuentes (fallback sin BERTopic).

        Utiliza extracción léxica simple: tokeniza los textos, identifica
        las palabras más frecuentes que aparecen en al menos el 5 % de los
        chunks, y asigna cada chunk al keyword que contenga. Los chunks
        sin keyword se asignan al tópico "otros".

        Args:
            texts: Lista de contenidos de chunks.

        Returns:
            Tupla (asignaciones, info_tópicos).
        """
        # ── Extraer keywords globales ────────────────────────────────
        keywords = self._extract_keywords(texts, top_n=15)

        if not keywords:
            logger.warning("No se encontraron keywords; agrupando todo como 'general'.")
            return (
                [0] * len(texts),
                {
                    0: {
                        "name": "general",
                        "description": "Corpus general (sin categorizar)",
                        "count": len(texts),
                    }
                },
            )

        # ── Asignar cada chunk al keyword más frecuente que contenga ──
        kw_order: list[str] = keywords  # preservar orden de frecuencia
        kw_to_topic: dict[str, int] = {kw: i for i, kw in enumerate(kw_order)}
        assignments: list[int] = []

        for text in texts:
            text_lower = text.lower()
            assigned = False
            for kw in kw_order:
                if kw in text_lower:
                    assignments.append(kw_to_topic[kw])
                    assigned = True
                    break
            if not assigned:
                assignments.append(-1)  # outlier → tópico "otros"

        # ── Construir info de tópicos ────────────────────────────────
        info: dict[int, dict[str, Any]] = {}

        for kw, tid in kw_to_topic.items():
            count = assignments.count(tid)
            if count > 0:
                info[tid] = {
                    "name": kw,
                    "description": f"Chunks relacionados con '{kw}'",
                    "count": count,
                }

        other_count = assignments.count(-1)
        if other_count > 0:
            info[-1] = {
                "name": "otros",
                "description": "Chunks no categorizados por keyword",
                "count": other_count,
            }

        logger.info(
            "Fallback completado: %d tópicos basados en keywords.",
            len(info),
        )
        return assignments, info

    @staticmethod
    def _extract_keywords(
        texts: list[str], top_n: int = 15
    ) -> list[str]:
        """Extrae las palabras clave más frecuentes del corpus.

        Tokeniza usando una expresión regular que captura palabras
        de al menos 4 caracteres, filtra stop words y selecciona
        aquellas que aparecen en al menos el 5 % de los chunks.

        Args:
            texts: Lista de contenidos de chunks.
            top_n: Número máximo de keywords a retornar.

        Returns:
            Lista de palabras clave ordenadas por frecuencia descendente.
        """
        word_counter: Counter[str] = Counter()
        token_pattern = re.compile(r"\b[a-záéíóúüñ]{4,}\b")

        for text in texts:
            words = token_pattern.findall(text.lower())
            for word in words:
                if word not in _ALL_STOP_WORDS:
                    word_counter[word] += 1

        # Filtrar palabras que aparecen en al menos el 5 % de los textos
        min_count = max(2, int(len(texts) * 0.05))
        keywords = [
            word
            for word, count in word_counter.most_common(top_n * 3)
            if count >= min_count
        ][:top_n]

        logger.debug(
            "%d keywords extraídas del corpus (min_count=%d).",
            len(keywords),
            min_count,
        )
        return keywords

    # ── Persistencia ─────────────────────────────────────────────────

    def _build_and_persist(
        self,
        assignments: list[int],
        topic_info: dict[int, dict[str, Any]],
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Agrupa chunks por tópico y persiste los resultados en PostgreSQL.

        Args:
            assignments: ID de tópico asignado a cada chunk (mismo orden).
            topic_info: Metadatos de cada tópico {topic_id: {name, description, count}}.
            chunk_ids: IDs de Qdrant de cada chunk (mismo orden).

        Returns:
            Lista de resúmenes de los tópicos persistidos, cada uno con
            'id', 'name', 'description', 'chunk_count' y 'representative_chunks'.
        """
        # ── Agrupar IDs de chunks por tópico ─────────────────────────
        topic_chunks: dict[int, list[str]] = {}
        for tid, cid in zip(assignments, chunk_ids, strict=True):
            topic_chunks.setdefault(tid, []).append(cid)

        # ── Limpiar tópicos anteriores ───────────────────────────────
        with Session(self._engine) as session:
            session.exec(delete(Topic))
            session.commit()
        logger.info("Tópicos anteriores eliminados de la base de datos.")

        # ── Persistir nuevos tópicos ─────────────────────────────────
        summaries: list[dict[str, Any]] = []

        with Session(self._engine) as session:
            for tid, info in topic_info.items():
                rep_chunks = topic_chunks.get(tid, [])[:MAX_REPRESENTATIVE_CHUNKS]
                count = info.get("count", len(rep_chunks))

                topic = Topic(
                    name=info["name"],
                    description=info.get("description", ""),
                    representative_chunks=rep_chunks,
                )
                session.add(topic)
                session.commit()
                session.refresh(topic)

                summaries.append(
                    {
                        "id": topic.id,
                        "name": topic.name,
                        "description": topic.description or "",
                        "chunk_count": count,
                        "representative_chunks": rep_chunks,
                    }
                )

        logger.info(
            "%d tópicos persistidos en la base de datos.", len(summaries)
        )
        return summaries


# ── Singleton ───────────────────────────────────────────────────────

_trainer: TopicTrainer | None = None


def get_topic_trainer() -> TopicTrainer:
    """Retorna la instancia singleton del entrenador de tópicos."""
    global _trainer
    if _trainer is None:
        _trainer = TopicTrainer()
    return _trainer
