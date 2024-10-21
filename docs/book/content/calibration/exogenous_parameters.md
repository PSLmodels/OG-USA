---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: ogusa-dev
---

(Chap_Exog)=
# Exogenous Parameters

  [TODO: This chapter needs heavy updating. Would be nice to do something similar to API chapter. But it is also nice to have references and descriptions as in the table below.]

  In this chapter, list the exogenous inputs to the model, options, and where the values come from (weak calibration vs. strong calibration). Point to the respective chapters for some of the inputs. Mention the code in [`default_parameters.json`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/default_parameters.json) and [`parameters.py`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/parameters.py).

  <!-- +++
  ```{code-cell} ogusa-dev
  :tags: [hide-cell]
  from myst_nb import glue
  import ogusa.parameter_tables as pt
  from ogusa import Specifications
  p = Specifications()
  table = pt.param_table(p, table_format=None, path=None)
  glue("param_table", table, display=False)
  ```
  -->

  **List of exogenous parameters and baseline calibration values.**
  | Symbol                           | Description                                                             | Value                                                 |
|:---------------------------------|:------------------------------------------------------------------------|:------------------------------------------------------|
| $\texttt{start\_year}$           | Initial year                                                            | 2025                                                  |
| $\omega_{s,t}$                   | Population by age over time                                             | Too large to report here, see default parameters JSON |
| $i_{s,t}$                        | Immigration rates by age                                                | Too large to report here, see default parameters JSON |
| $\rho_{s,t}$                     | Mortality rates by age                                                  | Too large to report here, see default parameters JSON |
| $e_{j,s,t}$                      | Deterministic ability process                                           | Too large to report here, see default parameters JSON |
| $\lambda_{j}$                    | Lifetime income group percentages                                       | Too large to report here, see default parameters JSON |
| $J$                              | Number of lifetime income groups                                        | 10                                                    |
| $S$                              | Maximum periods in economically active individual life                  | 80                                                    |
| $E$                              | Number of periods of youth economically outside the model               | 20                                                    |
| $T$                              | Number of periods to steady-state                                       | 320                                                   |
| $R$                              | Retirement age                                                          | [65.000...65.000]                                     |
| $\tilde{l}$                      | Maximum hours of labor supply                                           | 1.000                                                 |
| $\beta$                          | Discount factor                                                         | [0.910...0.995]                                       |
| $\sigma$                         | Coefficient of constant relative risk aversion                          | 1.500                                                 |
| $\nu$                            | Frisch elasticity of labor supply                                       | 0.400                                                 |
| $b$                              | Scale parameter in utility of leisure                                   | 0.573                                                 |
| $\upsilon$                       | Shape parameter in utility of leisure                                   | 2.856                                                 |
| $\chi^{n}_{s}$                   | Disutility of labor level parameters                                    | Too large to report here, see default parameters JSON |
| $\chi^{b}_{j}$                   | Utility of bequests level parameters                                    | [80.000...80.000]                                     |
| $\texttt{use\_zeta}$             | Whether to distribute bequests between lifetime income groups           | 0.00E+00                                              |
| $\zeta$                          | Distribution of bequests                                                | Too large to report here, see default parameters JSON |
| $Z_{t}$                          | Total factor productivity                                               | Too large to report here, see default parameters JSON |
| $\gamma$                         | Capital share of income                                                 | [0.380...0.380]                                       |
| $\varepsilon$                    | Elasticity of substitution between capital and labor                    | [1.000...1.000]                                       |
| $\delta$                         | Capital depreciation rate                                               | 0.050                                                 |
| $g_{y}$                          | Growth rate of labor augmenting technological progress                  | 0.020                                                 |
| $\texttt{tax\_func\_type}$       | Functional form used for income tax functions                           | DEP                                                   |
| $\texttt{analytical\_mtrs}$      | Whether use analytical MTRs or estimate MTRs                            | 0.00E+00                                              |
| $\texttt{age\_specific}$         | Whether use age-specific tax functions                                  | 1.000                                                 |
| $\tau^{p}_{t}$                   | Payroll tax rate                                                        | [0.000...0.000]                                       |
| $\tau^{BQ}_{t}$                  | Bequest (estate) tax rate                                               | [0.000...0.000]                                       |
| $\tau^{b}_{t}$                   | Entity-level business income tax rate                                   | Too large to report here, see default parameters JSON |
| $\delta^{\tau}_{t}$              | Rate of depreciation for tax purposes                                   | Too large to report here, see default parameters JSON |
| $\tau^{c}_{t,s,j}$               | Consumption tax rates                                                   | Too large to report here, see default parameters JSON |
| $H$                              | Coefficient on linear term in wealth tax function                       | [0.100...0.100]                                       |
| $M$                              | Constant in wealth tax function                                         | [1.000...1.000]                                       |
| $P$                              | Coefficient on level term in wealth tax function                        | [0.000...0.000]                                       |
| $\texttt{budget\_balance}$       | Whether have a balanced budget in each period                           | 0.00E+00                                              |
| $\texttt{baseline\_spending}$    | Whether level of spending constant between the baseline and reform runs | 0.00E+00                                              |
| $\alpha^{T}_{t}$                 | Transfers as a share of GDP                                             | [0.052...0.052]                                       |
| $\eta_{j,s,t}$                   | Distribution of transfers                                               | Too large to report here, see default parameters JSON |
| $\eta_{RM,j,s,t}$                | Distribution of remittances                                             | Too large to report here, see default parameters JSON |
| $\alpha^{G}_{t}$                 | Government spending as a share of GDP                                   | [0.091...0.091]                                       |
| $\alpha_{RM,1}$                  | Remittances as a share of GDP in initial period                         | 0.00E+00                                              |
| $\alpha_{RM,T}$                  | Remittances as a share of GDP in long run                               | 0.00E+00                                              |
| $g_{RM,t}$                       | Growth rate of remittances in initial periods                           | [0.000...0.000]                                       |
| $t_{G1}$                         | Model period in which budget closure rule starts                        | 20                                                    |
| $t_{G2}$                         | Model period in which budget closure rule ends                          | 256                                                   |
| $\rho_{G}$                       | Budget closure rule smoothing parameter                                 | 0.100                                                 |
| $\bar{\alpha}_{D}$               | Steady-state Debt-to-GDP ratio                                          | 2.000                                                 |
| $\alpha_{D,0}$                   | Initial period Debt-to-GDP ratio                                        | 0.990                                                 |
| $\tau_{d,t}$                     | Scale parameter in government interest rate wedge                       | [0.000...0.000]                                       |
| $\mu_{d,t}$                      | Shift parameter in government interest rate wedge                       | [-0.005...-0.010]                                     |
| $\texttt{avg\_earn\_num\_years}$ | Number of years over which compute average earnings for pension benefit | 35                                                    |
| $\texttt{AIME\_bkt\_1}$          | First AIME bracket threshold                                            | 1174.000                                              |
| $\texttt{AIME\_bkt\_2}$          | Second AIME bracket threshold                                           | 7078.000                                              |
| $\texttt{PIA\_rate\_bkt\_1}$     | First AIME bracket PIA rate                                             | 0.900                                                 |
| $\texttt{PIA\_rate\_bkt\_2}$     | Second AIME bracket PIA rate                                            | 0.320                                                 |
| $\texttt{PIA\_rate\_bkt\_3}$     | Third AIME bracket PIA rate                                             | 0.150                                                 |
| $\texttt{PIA\_maxpayment}$       | Maximum PIA payment                                                     | 4555.000                                              |
| $\texttt{PIA\_minpayment}$       | Minimum PIA payment                                                     | 0.00E+00                                              |
| $\theta_{adj,t}$                 | Adjustment to replacement rate                                          | [1.000...1.000]                                       |
| $r^{*}_{t}$                      | World interest rate                                                     | [0.040...0.040]                                       |
| $D_{f,0}$                        | Share of government debt held by foreigners in initial period           | 0.400                                                 |
| $\zeta_{D, t}$                   | Share of new debt issues purchased by foreigners                        | [0.400...0.400]                                       |
| $\zeta_{K, t}$                   | Share of excess capital demand satisfied by foreigners                  | [0.500...0.500]                                       |
| $\xi$                            | Dampening parameter for TPI                                             | 0.400                                                 |
| $\texttt{maxiter}$               | Maximum number of iterations for TPI                                    | 250                                                   |
| $\texttt{mindist\_SS}$           | SS solution tolerance                                                   | 1.00E-09                                              |
| $\texttt{mindist\_TPI}$          | TPI solution tolerance                                                  | 1.00E-05                                              |