"""
Background job manager for the OG-USA Panel app.

Runs the model in a daemon thread so the UI stays responsive during the
40–90 minute computation.  Uses ``param.Parameterized`` so Panel
components that watch ``status``, ``phase``, or ``elapsed`` update
reactively without any manual polling glue.

Email notification (optional) is sent via smtplib when the run completes
or fails.  Configure SMTP credentials through environment variables or
the UI's notification settings.
"""
import os
import smtplib
import threading
import time
import traceback
from email.mime.text import MIMEText

import param

from ogusa.app.backend.functions import run_model


# ---------------------------------------------------------------------------
# Optional SMTP settings (environment-variable driven)
# ---------------------------------------------------------------------------
_SMTP_HOST = os.environ.get("OGUSA_SMTP_HOST", "smtp.gmail.com")
_SMTP_PORT = int(os.environ.get("OGUSA_SMTP_PORT", "587"))
_SMTP_USER = os.environ.get("OGUSA_SMTP_USER", "")
_SMTP_PASS = os.environ.get("OGUSA_SMTP_PASS", "")
_FROM_ADDR = os.environ.get("OGUSA_FROM_ADDR", _SMTP_USER)


def _send_notification(to_addr: str, subject: str, body: str):
    """Send a plain-text email; silently skip if credentials are absent."""
    if not (to_addr and _SMTP_USER and _SMTP_PASS):
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = _FROM_ADDR
    msg["To"] = to_addr
    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(_SMTP_USER, _SMTP_PASS)
            server.sendmail(_FROM_ADDR, [to_addr], msg.as_string())
    except Exception as exc:  # noqa: BLE001
        print(f"[job_manager] Email notification failed: {exc}")


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class JobManager(param.Parameterized):
    """
    Reactive state container for one model run.

    Attributes
    ----------
    status : str
        ``"idle"`` | ``"running"`` | ``"done"`` | ``"error"``
    phase : str
        Human-readable description of the current computation step.
    elapsed : float
        Seconds since the run started.
    result : dict or None
        Output dict from ``comp_output()`` when status is ``"done"``.
    error_msg : str
        Traceback string when status is ``"error"``.
    """

    status = param.String(default="idle")
    phase = param.String(default="")
    elapsed = param.Number(default=0.0)
    result = param.Dict(default=None, allow_None=True)
    error_msg = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self._thread: threading.Thread | None = None
        self._timer: threading.Thread | None = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def run(
        self,
        meta_param_dict: dict,
        adjustment: dict,
        output_base: str,
        notify_email: str = "",
    ):
        """
        Launch the model in a background thread.

        Parameters
        ----------
        meta_param_dict, adjustment : dict
            As expected by ``backend.functions.run_model``.
        output_base : str
            Directory for baseline / reform output.
        notify_email : str
            If non-empty, send a completion/error email to this address.
        """
        if self.is_running:
            return

        self.status = "running"
        self.phase = "Initialising…"
        self.elapsed = 0.0
        self.result = None
        self.error_msg = ""
        self._start_time = time.monotonic()

        self._start_timer()

        self._thread = threading.Thread(
            target=self._worker,
            args=(meta_param_dict, adjustment, output_base, notify_email),
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_timer(self):
        """Tick elapsed time every second so the UI can show HH:MM:SS."""
        def _tick():
            while self.status == "running":
                time.sleep(1)
                self.elapsed = time.monotonic() - self._start_time

        self._timer = threading.Thread(target=_tick, daemon=True)
        self._timer.start()

    def _worker(
        self,
        meta_param_dict: dict,
        adjustment: dict,
        output_base: str,
        notify_email: str,
    ):
        try:
            result = run_model(
                meta_param_dict,
                adjustment,
                output_base,
                phase_callback=self._on_phase,
            )
            self.result = result
            self.status = "done"
            self.phase = "Complete"
            elapsed_fmt = _fmt_elapsed(self.elapsed)
            _send_notification(
                notify_email,
                "OG-USA run complete",
                f"Your OG-USA simulation finished in {elapsed_fmt}.\n\n"
                f"Output is saved to:\n{output_base}",
            )
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            self.error_msg = tb
            self.status = "error"
            self.phase = "Error"
            _send_notification(
                notify_email,
                "OG-USA run failed",
                f"Your OG-USA simulation encountered an error:\n\n{tb}",
            )

    def _on_phase(self, message: str):
        """Called by run_model between computation phases."""
        self.phase = message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"
