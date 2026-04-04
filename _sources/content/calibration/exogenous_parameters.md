---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(Chap_Exog)=
# Exogenous Parameters

  The JSON file [`ogusa_default_parameters.json`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/ogusa_default_parameters.json) provides values for all the model parameters used as defaults for OG-USA. Below, we provide a table highlighting some of the parameters describing the scale of the model (number of periods $T$, age groups $S$, productivity types $J$) and some parameters of the solution method (dampening parameter $\xi$ for TPI). The table below provides a list of the exogenous parameters and their baseline calibration values.


**Table: List of exogenous parameters and baseline calibration values**

```{include} ./images/exogenous_parameters_table.md
```
