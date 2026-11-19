"""Small standard-library HTTP client with timeouts, bounded retries, and size caps."""

from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    """Provider or download request failed."""


def get_json(url: str, headers: dict[str, str], timeout: int = 30, retries: int = 2) -> dict[str, Any]:
    if not url.startswith("https://"):
        raise ProviderError("Provider API requests must use HTTPS")
    if timeout <= 0 or retries < 0:
        raise ProviderError("Provider timeout must be positive and retries cannot be negative")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "UniversalVideoEditor/0.1", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > 10_000_000:
                    raise ProviderError("Provider response exceeds configured size limit")
                body = response.read(10_000_001)
                if len(body) > 10_000_000:
                    raise ProviderError("Provider response exceeds configured size limit")
                payload = json.loads(body.decode("utf-8"))
                if response.status >= 400 or not isinstance(payload, dict):
                    raise ProviderError(f"Invalid provider response (HTTP {response.status})")
                return payload
        except urllib.error.HTTPError as exc:
            if (exc.code == 429 or 500 <= exc.code < 600) and attempt < retries:
                time.sleep(min(2**attempt, 4))
                continue
            raise ProviderError(f"Provider request failed (HTTP {exc.code})") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
                continue
            raise ProviderError(f"Provider request failed: {type(exc).__name__}") from exc
    raise ProviderError("Provider request exhausted retries")


def _check_media_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or host not in allowed_hosts:
        raise ProviderError("Media download URL must be HTTPS on a permitted provider CDN")
    try:
        addresses = {row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ProviderError("Media CDN hostname could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ProviderError("Media CDN resolved to a non-public IP address")


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, hosts: set[str]) -> None:
        super().__init__()
        self.hosts = hosts

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        _check_media_url(newurl, self.hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(
    url: str, destination: Path, timeout: int, max_bytes: int, allowed_hosts: set[str] | None = None
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing downloaded media: {destination}")
    hosts = allowed_hosts or set()
    _check_media_url(url, hosts)
    opener = urllib.request.build_opener(_SafeRedirect(hosts))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    try:
        with (
            opener.open(
                urllib.request.Request(url, headers={"User-Agent": "UniversalVideoEditor/0.1"}), timeout=timeout
            ) as response,
            temporary.open("wb") as out,
        ):
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ProviderError("Media download exceeds configured size limit")
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ProviderError("Media download exceeds configured size limit")
                out.write(chunk)
        if temporary.stat().st_size == 0:
            raise ProviderError("Downloaded media file is empty")
        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
