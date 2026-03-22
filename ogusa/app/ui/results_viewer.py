"""
Renders the output dict from ``backend.functions.comp_output`` as Panel
components:
  - matplotlib Figure objects  →  pn.pane.Matplotlib
  - HTML table strings         →  pn.pane.HTML
  - CSV strings                →  pn.widgets.FileDownload (download button)
"""
import io

import panel as pn
import param


class ResultsViewer(param.Parameterized):
    """
    Builds a Panel layout from a ``comp_output`` result dict.

    Usage
    -----
    viewer = ResultsViewer()
    viewer.update(result_dict)   # call when a run finishes
    app_layout.append(viewer.panel())
    """

    _result: dict | None = None

    def update(self, result: dict):
        """Store a new result and trigger a UI refresh."""
        self._result = result
        self.param.trigger("name")  # lightweight way to notify watchers

    def panel(self) -> pn.viewable.Viewable:
        """Return the full results layout (figures + tables + downloads)."""
        if self._result is None:
            return pn.pane.Markdown(
                "*Results will appear here after a successful run.*",
                styles={"color": "#888"},
            )
        return _build_results_layout(self._result)


# ---------------------------------------------------------------------------
# Layout builder (module-level so it can be called independently)
# ---------------------------------------------------------------------------

def _build_results_layout(result: dict) -> pn.viewable.Viewable:
    """
    Convert a ``comp_output`` dict into a Panel column of:
      1. Figures in a row (or stacked if only one)
      2. HTML tables in a tab strip
      3. Download buttons
    """
    items = []

    # ---- Figures -----------------------------------------------------------
    figures = result.get("figures", [])
    if figures:
        items.append(pn.pane.Markdown("### Figures"))
        fig_panes = [
            pn.Column(
                pn.pane.Markdown(f"**{entry['title']}**"),
                pn.pane.Matplotlib(
                    entry["fig"],
                    tight=True,
                    sizing_mode="scale_both",
                    height=380,
                ),
            )
            for entry in figures
        ]
        if len(fig_panes) == 1:
            items.append(fig_panes[0])
        else:
            items.append(
                pn.GridBox(*fig_panes, ncols=2, sizing_mode="stretch_width")
            )

    # ---- Tables ------------------------------------------------------------
    tables = result.get("tables", [])
    if tables:
        items.append(pn.pane.Markdown("### Tables"))
        tab_entries = [
            (
                entry["title"],
                pn.pane.HTML(
                    _style_table(entry["html"]),
                    sizing_mode="stretch_width",
                ),
            )
            for entry in tables
        ]
        items.append(
            pn.Tabs(*tab_entries, dynamic=True, sizing_mode="stretch_width")
        )

    # ---- Downloads ---------------------------------------------------------
    downloads = result.get("downloads", [])
    if downloads:
        items.append(pn.pane.Markdown("### Downloads"))
        btns = []
        for entry in downloads:
            csv_bytes = entry["data"].encode("utf-8")
            btn = pn.widgets.FileDownload(
                filename=entry["filename"],
                callback=_make_csv_callback(csv_bytes),
                label=f"Download: {entry['title']}",
                button_type="success",
            )
            btns.append(btn)
        items.append(pn.Row(*btns))

    if not items:
        return pn.pane.Markdown("*No output to display.*")

    return pn.Column(*items, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _style_table(html: str) -> str:
    """Wrap the HTML table in a div with light styling."""
    style = (
        "<style>"
        "table { border-collapse: collapse; width: 100%; font-size: 0.9em; }"
        "th { background: #f0f4f8; padding: 6px 10px; text-align: left; "
        "     border-bottom: 2px solid #ccc; }"
        "td { padding: 5px 10px; border-bottom: 1px solid #eee; }"
        "tr:hover td { background: #fafafa; }"
        "</style>"
    )
    return f"{style}<div style='overflow-x:auto'>{html}</div>"


def _make_csv_callback(csv_bytes: bytes):
    """Return a zero-argument callback that yields a BytesIO for download."""
    def callback():
        return io.BytesIO(csv_bytes)
    return callback
