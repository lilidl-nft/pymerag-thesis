"""
Cliente LLM unificado para Pymerag.

Soporta llama.cpp local (API compatible con OpenAI) vía httpx,
con manejo robusto de errores y fallback informativo.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 1024
"""Tokens máximos por defecto en la respuesta generada."""

DEFAULT_TEMPERATURE = 0.3
"""Temperatura por defecto para generación determinista."""

DEFAULT_TIMEOUT = 60.0
"""Timeout en segundos para llamadas HTTP al LLM."""

SYSTEM_PROMPT = (
    "Eres un asistente de investigación especializado en análisis documental. "
    "Responde la pregunta del usuario basándote EXCLUSIVAMENTE en el contexto "
    "proporcionado. Si la información no está en el contexto, indícalo "
    "claramente. Cita las fuentes utilizadas."
)
"""Prompt de sistema que define el comportamiento del asistente RAG."""


class LLMClient:
    """Cliente para interactuar con un LLM a través de API compatible con OpenAI.

    Soporta tanto servidores locales (llama.cpp) como servicios cloud
    a través del mismo endpoint configurado en settings.

    Attributes:
        base_url: URL base del endpoint (ej. http://localhost:8080/v1).
        model: Nombre del modelo a usar en las llamadas a la API.
        timeout: Timeout HTTP en segundos.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "deepseek-v4",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Inicializa el cliente LLM.

        Args:
            base_url: URL base del endpoint LLM (usa settings si es None).
            model: Nombre del modelo.
            timeout: Timeout HTTP en segundos.
        """
        self.base_url = base_url or settings.llm_api_base
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Genera una respuesta del LLM basada en el prompt y contexto.

        Construye el mensaje completo con system prompt, contexto
        recuperado y la consulta del usuario, y llama a la API
        compatible con OpenAI.

        Args:
            prompt: Consulta o instrucción del usuario.
            context: Lista de fragmentos de texto como contexto.
            max_tokens: Número máximo de tokens en la respuesta.
            temperature: Temperatura para muestreo (0.0 = determinista).

        Returns:
            Texto de la respuesta generada por el LLM.
        """
        messages = self._build_messages(prompt, context)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.ConnectError:
            logger.warning(
                "No se pudo conectar al LLM en %s. "
                "Verifica que el servidor esté ejecutándose.",
                self.base_url,
            )
            return self._fallback_response(context)

        except httpx.TimeoutException:
            logger.warning(
                "Timeout esperando respuesta del LLM en %s (timeout=%.1fs).",
                self.base_url,
                self.timeout,
            )
            return self._fallback_response(context)

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "El LLM retornó error HTTP %d: %s.",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return self._fallback_response(context)

        except Exception:
            logger.exception("Error inesperado en generación LLM.")
            return (
                "[Error interno del sistema al generar la respuesta. "
                "Consulte los fragmentos recuperados a continuación.]\n\n"
                + self._format_context(context)
            )

    async def agenerate(
        self,
        prompt: str,
        context: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Versión asíncrona de generate().

        Args:
            prompt: Consulta o instrucción del usuario.
            context: Lista de fragmentos de texto como contexto.
            max_tokens: Número máximo de tokens en la respuesta.
            temperature: Temperatura para muestreo.

        Returns:
            Texto de la respuesta generada por el LLM.
        """
        messages = self._build_messages(prompt, context)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.ConnectError:
            logger.warning(
                "No se pudo conectar al LLM en %s (async).", self.base_url
            )
            return self._fallback_response(context)

        except httpx.TimeoutException:
            logger.warning(
                "Timeout en generación asíncrona (timeout=%.1fs).",
                self.timeout,
            )
            return self._fallback_response(context)

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "El LLM retornó error HTTP %d (async): %s.",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return self._fallback_response(context)

        except Exception:
            logger.exception("Error inesperado en generación LLM asíncrona.")
            return (
                "[Error interno del sistema al generar la respuesta.]\n\n"
                + self._format_context(context)
            )

    def _build_messages(
        self,
        prompt: str,
        context: list[str] | None,
    ) -> list[dict[str, str]]:
        """Construye la lista de mensajes para la API de chat.

        Args:
            prompt: Consulta del usuario.
            context: Fragmentos de contexto recuperados.

        Returns:
            Lista de mensajes en formato OpenAI (role + content).
        """
        system_content = SYSTEM_PROMPT

        if context:
            context_text = self._format_context(context)
            system_content += f"\n\n## Contexto proporcionado\n{context_text}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _format_context(context: list[str] | None) -> str:
        """Formatea los fragmentos de contexto para incluir en el prompt.

        Args:
            context: Lista de textos de contexto.

        Returns:
            Texto formateado con fuentes numeradas.
        """
        if not context:
            return "[No se encontró contexto relevante.]"

        parts: list[str] = []
        for i, chunk in enumerate(context, 1):
            parts.append(f"[Fuente {i}]\n{chunk}")

        return "\n\n".join(parts)

    @staticmethod
    def _fallback_response(context: list[str] | None) -> str:
        """Genera una respuesta de fallback cuando el LLM no está disponible.

        Args:
            context: Fragmentos de contexto recuperados.

        Returns:
            Mensaje informativo con los fragmentos recuperados.
        """
        header = (
            "⚠️ No se pudo conectar con el modelo de lenguaje. "
            "A continuación se muestran los fragmentos recuperados "
            "que podrían ser relevantes para tu consulta:\n\n"
        )
        context_text = LLMClient._format_context(context)
        return header + context_text


# ── Singleton ────────────────────────────────────────────────────────
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Retorna la instancia singleton del cliente LLM."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
