"""
OG-USA Panel web application.

Launch locally:
    panel serve ogusa/app/app.py --show

Or from the repo root with the helper script:
    ./launch_ogusa.command
"""
import os
import pathlib
import sys
import time

# When served with `panel serve ogusa/app/app.py`, this file is executed
# as a script (not imported as a package), so relative imports fail.
# Adding the repo root to sys.path makes the absolute imports below work
# in both the panel-serve and the package-import contexts.
_REPO_ROOT = str(pathlib.Path(__file__).parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import panel as pn
import param

from ogusa.app.backend.functions import MetaParams, get_inputs, get_version
from ogusa.app.ui.job_manager import JobManager, _fmt_elapsed
from ogusa.app.ui.parameter_form import ParameterForm
from ogusa.app.ui.results_viewer import ResultsViewer, _build_results_layout

pn.extension(
    "tabulator",
    sizing_mode="stretch_width",
    notifications=True,
)

# ---------------------------------------------------------------------------
# Default output directory: ~/.ogusa_app/runs/<timestamp>
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_ROOT = pathlib.Path.home() / ".ogusa_app" / "runs"


def _make_output_dir() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = _DEFAULT_OUTPUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ---------------------------------------------------------------------------
# OGUSAApp – top-level param.Parameterized class
# ---------------------------------------------------------------------------

class OGUSAApp(param.Parameterized):
    """Holds shared state and wires the UI components together."""

    # Meta-parameter widgets (sidebar)
    start_year = param.Integer(
        default=2021,
        bounds=(2015, 2035),
        label="Start Year",
        doc="First year of the budget window.",
    )
    data_source = param.ObjectSelector(
        default="CPS",
        objects=["CPS", "PUF"],
        label="Data Source",
    )
    time_path = param.Boolean(
        default=True,
        label="Solve Transition Path",
        doc=(
            "Solve the full time-path equilibrium in addition to the "
            "steady state.  Adds ~30 min of compute time."
        ),
    )
    notify_email = param.String(
        default="",
        label="Notify email (optional)",
        doc="Receive an email when the run finishes.",
    )

    def __init__(self, **params):
        super().__init__(**params)

        # Load default parameter specs from ParamTools
        self._inputs = self._load_inputs()

        # UI components
        self._form = ParameterForm(self._inputs)
        self._job = JobManager()
        self._viewer = ResultsViewer()

        # Watch job status to refresh Results tab
        self._job.param.watch(self._on_status_change, "status")

        # Sidebar widgets
        self._run_btn = pn.widgets.Button(
            name="Run Model",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self._run_btn.on_click(self._on_run_click)

        self._status_md = pn.pane.Markdown(
            self._status_text(),
            sizing_mode="stretch_width",
        )
        self._phase_md = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
            styles={"color": "#555", "font-size": "0.85em"},
        )

        # Periodic callback to refresh elapsed time while running
        pn.state.add_periodic_callback(self._tick, period=1000)

    # ------------------------------------------------------------------
    # Input loading
    # ------------------------------------------------------------------

    def _load_inputs(self) -> dict:
        meta_dict = {
            "year": [{"value": self.start_year}],
            "data_source": [{"value": self.data_source}],
            "time_path": [{"value": self.time_path}],
        }
        return get_inputs(meta_dict)

    def _meta_param_dict(self) -> dict:
        return {
            "year": [{"value": self.start_year}],
            "data_source": [{"value": self.data_source}],
            "time_path": [{"value": self.time_path}],
        }

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_run_click(self, event):
        if self._job.is_running:
            pn.state.notifications.warning(
                "A run is already in progress.", duration=3000
            )
            return

        # Validate TC JSON before submitting
        try:
            tc_adj = self._form.get_tc_adjustments()
        except ValueError as exc:
            pn.state.notifications.error(str(exc), duration=6000)
            return

        adjustment = {
            "OG-USA Parameters": self._form.get_ogusa_adjustments(),
            "Tax-Calculator Parameters": tc_adj,
        }

        output_base = _make_output_dir()
        self._run_btn.disabled = True
        self._job.run(
            self._meta_param_dict(),
            adjustment,
            output_base,
            notify_email=self.notify_email,
        )
        self._refresh_status()

    def _on_status_change(self, event):
        self._refresh_status()
        if event.new == "done" and self._job.result:
            self._viewer.update(self._job.result)
            self._run_btn.disabled = False
            pn.state.notifications.success(
                "Run complete! See the Results tab.", duration=5000
            )
        elif event.new == "error":
            self._run_btn.disabled = False
            pn.state.notifications.error(
                "Run failed. See the Status panel for details.",
                duration=8000,
            )

    def _tick(self):
        """Refresh elapsed display every second while running."""
        if self._job.is_running:
            self._refresh_status()

    def _refresh_status(self):
        self._status_md.object = self._status_text()
        self._phase_md.object = (
            f"*{self._job.phase}*" if self._job.phase else ""
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _status_text(self) -> str:
        status = self._job.status
        elapsed = _fmt_elapsed(self._job.elapsed)
        icons = {
            "idle": "⚪",
            "running": "🔵",
            "done": "🟢",
            "error": "🔴",
        }
        icon = icons.get(status, "⚪")
        labels = {
            "idle": "Ready",
            "running": f"Running — {elapsed}",
            "done": f"Done ({elapsed})",
            "error": "Failed",
        }
        label = labels.get(status, status.title())
        return f"**Status:** {icon} {label}"

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def sidebar(self) -> pn.viewable.Viewable:
        meta_widgets = pn.param.Param(
            self,
            parameters=["start_year", "data_source", "time_path", "notify_email"],
            show_name=False,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            pn.pane.Markdown(f"### OG-USA v{get_version()}"),
            pn.layout.Divider(),
            pn.pane.Markdown("#### Simulation Settings"),
            meta_widgets,
            pn.layout.Divider(),
            self._run_btn,
            self._status_md,
            self._phase_md,
            sizing_mode="stretch_width",
        )

    def main(self) -> pn.viewable.Viewable:
        # Results tab content (reactive: re-renders when viewer updates)
        @pn.depends(self._viewer.param.name)
        def results_content(_):
            if self._job.status == "error":
                return pn.Column(
                    pn.pane.Markdown("### Run Error"),
                    pn.pane.Markdown(
                        f"```\n{self._job.error_msg}\n```",
                        sizing_mode="stretch_width",
                    ),
                )
            return _build_results_layout(self._job.result) if self._job.result else pn.pane.Markdown(
                "*Results will appear here after a successful run.*",
                styles={"color": "#888"},
            )

        return pn.Tabs(
            (
                "OG-USA Parameters",
                pn.Column(
                    pn.pane.Markdown(
                        "Adjust model parameters below.  "
                        "Leave unchanged to use OG-USA defaults."
                    ),
                    self._form.ogusa_panel(),
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Tax-Calculator",
                pn.Column(
                    self._form.tc_panel(),
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Results",
                pn.Column(
                    results_content,
                    sizing_mode="stretch_width",
                ),
            ),
            dynamic=True,
            sizing_mode="stretch_width",
        )


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

def create_app():
    app = OGUSAApp()
    template = pn.template.FastListTemplate(
        title="OG-USA: Overlapping Generations Model for the US",
        sidebar=[app.sidebar()],
        main=[app.main()],
        theme="default",
        accent_base_color="#1a6496",
        header_background="#1a6496",
    )
    return template


# Make it servable (used by `panel serve`)
create_app().servable()
