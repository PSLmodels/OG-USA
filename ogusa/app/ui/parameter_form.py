"""
Auto-generate Panel widgets from a ParamTools Specifications dump.

OG-USA parameters get fully typed widgets (float/int inputs, checkboxes,
selects).  Array-valued parameters get a JSON text area.

Tax-Calculator reforms are handled separately as a plain JSON text area
so users can paste any TC-compatible reform dict without needing a widget
for every one of the 200+ TC parameters.
"""
import json
from collections import OrderedDict

import panel as pn
import param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_default(param_meta: dict):
    """Return the scalar or list default from a ParamTools value entry."""
    values = param_meta.get("value", [])
    if not values:
        return None
    return values[0].get("value")


def _make_widget(name: str, param_meta: dict):
    """
    Build a Panel widget appropriate for one OG-USA parameter.

    Returns
    -------
    panel widget
    """
    title = param_meta.get("title") or name
    description = param_meta.get("description", "")
    ptype = param_meta.get("type", "float")
    validators = param_meta.get("validators", {})
    default = _extract_default(param_meta)

    # Array parameters → editable JSON textarea
    if isinstance(default, list):
        return pn.widgets.TextAreaInput(
            name=title,
            value=json.dumps(default),
            placeholder="JSON array",
            rows=3,
            sizing_mode="stretch_width",
        )

    # Boolean
    if ptype == "bool":
        return pn.widgets.Checkbox(
            name=title,
            value=bool(default),
        )

    # String with enumerated choices
    if "choice" in validators:
        choices = validators["choice"].get("choices", [])
        return pn.widgets.Select(
            name=title,
            value=str(default) if default is not None else choices[0],
            options=choices,
        )

    # Numeric range bounds – some validators reference other parameter
    # names (e.g. "max": "tG2") rather than literals; treat those as
    # unbounded.
    range_v = validators.get("range", {})

    def _to_num(v, cast):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    lo = _to_num(range_v.get("min"), float)
    hi = _to_num(range_v.get("max"), float)

    if ptype == "int":
        return pn.widgets.IntInput(
            name=title,
            value=int(default) if default is not None else 0,
            start=int(lo) if lo is not None else None,
            end=int(hi) if hi is not None else None,
            step=1,
            sizing_mode="stretch_width",
        )

    # Default: float
    return pn.widgets.FloatInput(
        name=title,
        value=float(default) if default is not None else 0.0,
        start=lo,
        end=hi,
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# ParameterForm
# ---------------------------------------------------------------------------

class ParameterForm(param.Parameterized):
    """
    Builds and manages all user-facing parameter widgets.

    Parameters
    ----------
    inputs_dict : dict
        The dict returned by ``backend.functions.get_inputs()``.
    """

    def __init__(self, inputs_dict: dict, **params):
        super().__init__(**params)
        self._inputs = inputs_dict
        self._ogusa_widgets: dict[str, pn.widgets.Widget] = {}
        self._tc_reform_area: pn.widgets.TextAreaInput | None = None
        self._build_ogusa_widgets()
        self._build_tc_widget()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ogusa_widgets(self):
        ogusa_params = self._inputs["model_parameters"].get(
            "OG-USA Parameters", {}
        )
        for name, meta in ogusa_params.items():
            if name == "schema":
                continue
            self._ogusa_widgets[name] = _make_widget(name, meta)

    def _build_tc_widget(self):
        self._tc_reform_area = pn.widgets.TextAreaInput(
            name="Tax-Calculator Reform (JSON)",
            value="{}",
            placeholder=(
                '{\n  "II_rt1": {"2025": 0.09},\n'
                '  "SS_Earnings_c": {"2025": 200000}\n}'
            ),
            rows=12,
            sizing_mode="stretch_width",
            description=(
                "Paste a Tax-Calculator reform dict here.  Keys are "
                "parameter names from Tax-Calculator's "
                "policy_current_law.json; values are year-keyed dicts. "
                "Leave as {} for current law."
            ),
        )

    # ------------------------------------------------------------------
    # Value extraction
    # ------------------------------------------------------------------

    def get_ogusa_adjustments(self) -> dict:
        """
        Return a dict of OG-USA parameter values that differ from their
        defaults, ready to pass into ``run_model``.
        """
        ogusa_params = self._inputs["model_parameters"].get(
            "OG-USA Parameters", {}
        )
        adjustments = {}
        for name, widget in self._ogusa_widgets.items():
            meta = ogusa_params.get(name, {})
            default = _extract_default(meta)
            val = widget.value

            # JSON textarea → parse back to list
            if isinstance(widget, pn.widgets.TextAreaInput):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    continue  # skip malformed JSON silently

            if val != default:
                adjustments[name] = val

        return adjustments

    def get_tc_adjustments(self) -> dict:
        """
        Return the parsed Tax-Calculator reform dict from the text area.
        Returns an empty dict if the field is blank or contains only {}.
        """
        raw = (self._tc_reform_area.value or "").strip()
        if not raw or raw == "{}":
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in Tax-Calculator reform field: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def ogusa_panel(self) -> pn.viewable.Viewable:
        """
        Return a tabbed Panel layout of all OG-USA parameter widgets,
        organised by section_1 → section_2.
        """
        ogusa_params = self._inputs["model_parameters"].get(
            "OG-USA Parameters", {}
        )

        # Build section map:  section_1 → section_2 → [param_names]
        sections: dict[str, dict[str, list[str]]] = OrderedDict()
        for name, meta in ogusa_params.items():
            if name == "schema" or name not in self._ogusa_widgets:
                continue
            s1 = meta.get("section_1") or "Other"
            s2 = meta.get("section_2") or ""
            sections.setdefault(s1, OrderedDict()).setdefault(
                s2, []
            ).append(name)

        tab_contents = []
        for s1, subsections in sections.items():
            col_items = []
            for s2, names in subsections.items():
                if s2:
                    col_items.append(
                        pn.pane.Markdown(
                            f"**{s2}**",
                            margin=(10, 0, 4, 0),
                        )
                    )
                for name in names:
                    widget = self._ogusa_widgets[name]
                    meta = ogusa_params[name]
                    desc = meta.get("description", "")
                    col_items.append(
                        pn.Column(
                            widget,
                            pn.pane.Markdown(
                                f"*{desc[:120]}*" if desc else "",
                                styles={
                                    "font-size": "0.78em",
                                    "color": "#666",
                                },
                                margin=(0, 0, 6, 0),
                            ),
                            margin=(0, 0, 4, 0),
                        )
                    )

            tab_contents.append(
                (
                    s1,
                    pn.Column(
                        *col_items,
                        scroll=True,
                        sizing_mode="stretch_width",
                        height=520,
                    ),
                )
            )

        if not tab_contents:
            return pn.pane.Markdown("*No adjustable parameters found.*")

        return pn.Tabs(*tab_contents, dynamic=True, sizing_mode="stretch_width")

    def tc_panel(self) -> pn.viewable.Viewable:
        """Return the Tax-Calculator reform JSON input panel."""
        return pn.Column(
            pn.pane.Markdown(
                "### Tax-Calculator Reform\n\n"
                "Enter a Tax-Calculator reform as a JSON object.  "
                "Parameter names and valid value ranges are documented in "
                "[Tax-Calculator's parameter index]"
                "(https://pslmodels.github.io/Tax-Calculator/index.html).  "
                "Leave as `{}` to use current law (no reform)."
            ),
            self._tc_reform_area,
            sizing_mode="stretch_width",
        )
