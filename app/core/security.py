"""
Módulo de seguridad: JWT, RBAC, hashing de contraseñas y rate limiting.

Proporciona las dependencias de FastAPI para autenticación, autorización
basada en roles, manejo de contraseñas con bcrypt, y un limitador de
tasa en memoria para proteger los endpoints de la API.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Esquema de autenticación ──────────────────────────────────────────
security_scheme = HTTPBearer(
    scheme_name="Bearer",
    description="Token JWT obtenido en /api/v1/auth/login",
)

# ── Constantes ────────────────────────────────────────────────────────
ALGORITHM = settings.jwt_algorithm
"""Algoritmo de firma JWT (HS256 por defecto)."""

SECRET_KEY = settings.jwt_secret
"""Clave secreta para firmar y verificar tokens JWT."""

ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_expire_minutes
"""Duración por defecto de los tokens de acceso."""


# ═══════════════════════════════════════════════════════════════════════
# JWT — Creación y validación de tokens
# ═══════════════════════════════════════════════════════════════════════


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Crea un token JWT de acceso firmado.

    Args:
        data: Diccionario con los claims a incluir en el token
              (típicamente 'sub' para el user_id y 'role' para el rol).
        expires_delta: Tiempo de expiración personalizado.
                       Si es None, usa el valor por defecto de settings.

    Returns:
        Token JWT codificado como string.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
        }
    )

    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug("Token JWT creado para subject=%s", data.get("sub", "?"))
    return token


def decode_token(token: str) -> dict[str, Any]:
    """Decodifica y verifica un token JWT.

    Args:
        token: Token JWT a decodificar.

    Returns:
        Diccionario con los claims del token.

    Raises:
        jwt.ExpiredSignatureError: Si el token ha expirado.
        jwt.InvalidTokenError: Si el token es inválido.
    """
    payload: dict[str, Any] = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM],
        options={"require": ["exp", "iat", "sub"]},
    )
    return payload


# ═══════════════════════════════════════════════════════════════════════
# FastAPI Dependencies — Autenticación y autorización
# ═══════════════════════════════════════════════════════════════════════


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> dict[str, Any]:
    """Dependencia de FastAPI: extrae y valida el usuario desde el token JWT.

    Se inyecta en los endpoints protegidos. Decodifica el token del header
    Authorization: Bearer <token> y retorna los claims del usuario.

    Args:
        credentials: Credenciales extraídas del header Authorization.

    Returns:
        Diccionario con los claims del token (sub, role, exp, etc.).

    Raises:
        HTTPException 401: Si el token es inválido o ha expirado.
    """
    token = credentials.credentials

    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        logger.warning("Token JWT expirado.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expirado. Inicie sesión nuevamente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("Token JWT inválido: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verificar que tenga los claims mínimos
    if "sub" not in payload:
        logger.warning("Token JWT sin claim 'sub'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token malformado: falta identidad del usuario.",
        )

    return payload


def require_role(required_role: str):
    """Factory de dependencia FastAPI para control de acceso basado en roles (RBAC).

    Retorna una dependencia que verifica que el usuario autenticado tenga
    el rol requerido. Se usa encadenada después de get_current_user.

    Uso:
        @router.get("/admin")
        async def admin_endpoint(
            user: dict = Depends(get_current_user),
            _: None = Depends(require_role("admin")),
        ):
            ...

    Args:
        required_role: Nombre del rol requerido (ej. 'admin', 'editor').

    Returns:
        Función de dependencia que lanza HTTPException 403 si el rol no coincide.
    """

    async def role_checker(
        user: dict[str, Any] = Depends(get_current_user),
    ) -> None:
        user_role = user.get("role", "")
        if user_role != required_role:
            logger.warning(
                "Acceso denegado: usuario=%s requiere rol=%s, tiene rol=%s",
                user.get("sub", "?"),
                required_role,
                user_role,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Se requiere el rol '{required_role}' para acceder a este recurso.",
            )

    return role_checker


# ═══════════════════════════════════════════════════════════════════════
# Hashing de contraseñas con bcrypt
# ═══════════════════════════════════════════════════════════════════════


def hash_password(password: str) -> str:
    """Genera un hash bcrypt de la contraseña.

    Args:
        password: Contraseña en texto plano.

    Returns:
        Hash bcrypt como string (incluye el salt).
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica una contraseña en texto plano contra un hash bcrypt.

    Args:
        plain_password: Contraseña en texto plano proporcionada por el usuario.
        hashed_password: Hash bcrypt almacenado.

    Returns:
        True si la contraseña coincide, False en caso contrario.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ═══════════════════════════════════════════════════════════════════════
# Rate Limiter en memoria (sliding window)
# ═══════════════════════════════════════════════════════════════════════


class RateLimiter:
    """Limitador de tasa en memoria con ventana deslizante simple.

    Rastrea las solicitudes por clave (típicamente IP o user_id) y
    rechaza aquellas que excedan el límite configurado dentro de la
    ventana de tiempo.

    Attributes:
        max_requests: Número máximo de solicitudes permitidas por ventana.
        window_seconds: Duración de la ventana en segundos.
    """

    def __init__(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        """Inicializa el rate limiter.

        Args:
            max_requests: Límite de solicitudes por ventana
                          (usa settings.rate_limit_requests si es None).
            window_seconds: Duración de la ventana en segundos
                            (usa settings.rate_limit_window si es None).
        """
        self.max_requests = max_requests or settings.rate_limit_requests
        self.window_seconds = window_seconds or settings.rate_limit_window
        self._store: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Verifica si una solicitud está permitida para la clave dada.

        Args:
            key: Identificador del cliente (IP, user_id, etc.).

        Returns:
            True si la solicitud está dentro del límite, False si fue rate-limited.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        # Limpiar timestamps antiguos
        timestamps = self._store[key]
        self._store[key] = [ts for ts in timestamps if ts > window_start]

        if len(self._store[key]) >= self.max_requests:
            logger.warning(
                "Rate limit alcanzado para key=%s (%d req en %ds).",
                key,
                self.max_requests,
                self.window_seconds,
            )
            return False

        self._store[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        """Retorna el número de solicitudes restantes en la ventana actual.

        Args:
            key: Identificador del cliente.

        Returns:
            Número de solicitudes que aún se pueden realizar.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds
        timestamps = self._store.get(key, [])
        active = [ts for ts in timestamps if ts > window_start]
        return max(0, self.max_requests - len(active))

    def reset(self, key: str) -> None:
        """Reinicia el contador para una clave específica.

        Args:
            key: Identificador del cliente a reiniciar.
        """
        self._store.pop(key, None)
        logger.debug("Rate limit reseteado para key=%s.", key)

    def cleanup(self) -> int:
        """Elimina las entradas expiradas del almacén interno.

        Útil para evitar crecimiento ilimitado de memoria en procesos
        de larga duración con muchas claves efímeras.

        Returns:
            Número de claves eliminadas.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        keys_to_remove: list[str] = []
        for key, timestamps in self._store.items():
            active = [ts for ts in timestamps if ts > window_start]
            if not active:
                keys_to_remove.append(key)
            else:
                self._store[key] = active

        for key in keys_to_remove:
            del self._store[key]

        if keys_to_remove:
            logger.debug(
                "Limpieza de rate limiter: %d claves eliminadas.",
                len(keys_to_remove),
            )
        return len(keys_to_remove)


# ── Singleton ────────────────────────────────────────────────────────
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Retorna la instancia singleton del rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# ── Dependencia FastAPI para rate limiting ───────────────────────────


async def rate_limit_dependency(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """Dependencia de FastAPI que aplica rate limiting por IP del cliente.

    Se inyecta en los endpoints que requieren protección contra abuso.
    Usa la dirección IP del cliente como clave de rate limiting.

    Args:
        request: Request de FastAPI (auto-inyectado).
        limiter: Instancia del rate limiter (singleton).

    Raises:
        HTTPException 429: Si se excede el límite de solicitudes.
    """
    client_ip = request.client.host if request.client else "unknown"

    if not limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Demasiadas solicitudes. Intente nuevamente más tarde.",
        )
