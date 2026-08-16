"""
Detects where training code is running and prints/returns the right
clickable "live monitoring" link for the prc dashboard - the same link
regardless of whether you're in a local script, Jupyter, Google Colab,
Kaggle, or on a remote/SSH box.

This never raises: if detection or link construction fails for any
reason, we fall back to a plain localhost URL rather than crash the
caller's training process.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("prc")


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _is_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or os.environ.get("KAGGLE_URL_BASE") is not None


def _try_ngrok_tunnel(port: int) -> Optional[str]:
    """
    Best-effort ngrok tunnel, used as a fallback wherever there's no
    built-in proxy (Kaggle, remote/SSH boxes, plain servers you want to
    share externally). Opt-in only: requires `pip install pyngrok` and,
    for anything beyond the free tier's short-lived tunnels, an
    NGROK_AUTHTOKEN env var. Silently returns None if pyngrok isn't
    installed or the tunnel can't be created - this is a nice-to-have,
    never a requirement for prc to work.
    """
    try:
        from pyngrok import ngrok, conf  # type: ignore
    except ImportError:
        return None

    try:
        token = os.environ.get("NGROK_AUTHTOKEN")
        if token:
            conf.get_default().auth_token = token
        tunnel = ngrok.connect(port, "http")
        return str(tunnel.public_url)
    except Exception:
        logger.debug("prc: ngrok tunnel failed (non-fatal)", exc_info=True)
        return None


def _is_ssh_session() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def detect_environment() -> str:
    """Returns one of: 'colab', 'kaggle', 'ssh', 'jupyter', 'local'."""
    if _is_colab():
        return "colab"
    if _is_kaggle():
        return "kaggle"
    if _is_ssh_session():
        return "ssh"
    if _is_jupyter():
        return "jupyter"
    return "local"


def _colab_proxy_url(port: int, path: str) -> Optional[str]:
    try:
        from google.colab.output import eval_js  # type: ignore
        base = eval_js(f"google.colab.kernel.proxyPort({port})")
        return base.rstrip("/") + path
    except Exception:
        logger.debug("prc: colab proxy URL construction failed", exc_info=True)
        return None


def build_live_url(server_url: str, run_id: str) -> tuple[str, str]:
    """
    Returns (url, note). `note` is an optional extra line of guidance
    (e.g. "you may need to port-forward") - empty string if none needed.
    """
    parsed = urlparse(server_url)
    port = parsed.port or 8000
    path = f"/runs/{run_id}"
    env = detect_environment()

    if env == "colab":
        url = _colab_proxy_url(port, path)
        if url:
            return url, ""
        # Fall through to local URL if the proxy call failed for any reason.
        return f"{server_url}{path}", "Could not auto-detect a Colab proxy URL; if this link doesn't load, re-run this cell."

    if env == "kaggle":
        # Kaggle doesn't expose a general-purpose arbitrary-port proxy the
        # way Colab does. Try an opt-in ngrok tunnel first; fall back to
        # the local URL with a clear explanation if that's unavailable.
        tunnel_url = _try_ngrok_tunnel(port)
        if tunnel_url:
            return f"{tunnel_url}{path}", "Using a temporary ngrok tunnel since Kaggle doesn't proxy arbitrary ports."
        return (
            f"{server_url}{path}",
            "Kaggle doesn't support automatic port proxying for arbitrary ports, so this "
            "link likely won't load from the notebook. Run `pip install pyngrok` and set "
            "NGROK_AUTHTOKEN to get a working tunnel automatically, or run the prc server "
            "outside Kaggle.",
        )

    if env == "ssh":
        tunnel_url = _try_ngrok_tunnel(port)
        if tunnel_url:
            return f"{tunnel_url}{path}", "Using a temporary ngrok tunnel."
        return (
            f"{server_url}{path}",
            f"You're on a remote/SSH session - forward the port to view this locally, "
            f"e.g.: ssh -L {port}:localhost:{port} <this-host> "
            f"(or `pip install pyngrok` for an automatic tunnel next time)",
        )

    return f"{server_url}{path}", ""


def announce_live_url(server_url: str, run_id: str) -> Optional[str]:
    """Fail-safe: prints the live monitoring link and returns it. Never
    raises - returns None if anything goes wrong."""
    try:
        url, note = build_live_url(server_url, run_id)
        print(f"\U0001F534 Live monitoring: {url}")
        if note:
            print(f"   note: {note}")

        # In notebook environments, also render a clickable HTML link
        # since plain printed URLs aren't always auto-linkified.
        try:
            from IPython.display import display, HTML
            if detect_environment() in ("colab", "kaggle", "jupyter"):
                display(HTML(f'<a href="{url}" target="_blank">Open live monitoring &rarr;</a>'))
        except Exception:
            pass

        return url
    except Exception:
        logger.exception("prc: failed to announce live URL (non-fatal)")
        return None
