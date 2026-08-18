
#!/bin/Python3

import pandas as pa
import numpy as np
import scipy.stats as stats
import growthcurves as gc

"""
Assisted by Claude AI with correction of related equations and coding, using references below (!!!! Add references !!!!)
Parametric microbial growth models.

All models share the same signature:

    model(t, N0, K, mu, lag)  →  (N(t), t_inflection)

Parameters
----------
t   : float or np.ndarray  – time
N0  : float                – initial population (value at t = lag for
                             logistic / Richards; value at t = 0 for
                             Gompertz / Baranyi)
K   : float                – carrying capacity (upper asymptote), K > N0
mu  : float                – maximum specific growth rate  [1/time]
lag : float                – lag phase duration  [time]

Returns
-------
growth        : float or np.ndarray  – population N at time t
t_inflection  : float                – time of maximum growth rate dN/dt
"""

# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------

def logistic_growth(t, N0, K, mu, lag):
    """
    Logistic growth model with smooth transition through lag phase.

    The standard logistic ODE  dN/dt = µ · N · (1 − N/K)  is solved
    analytically, then shifted so that the sigmoid is centred at t = lag
    (i.e. N(lag) = N0):

        N(t) = K / (1 + C · exp(−µ · (t − lag)))

    where  C = (K − N0) / N0.

    The inflection (maximum growth rate, N = K/2) is at:

        t_inf = lag + ln((K − N0) / N0) / µ

    Reference
    ---------
    Verhulst PF (1838). Notice sur la loi que la population suit dans
    son accroissement. Correspondance Mathématique et Physique, 10, 113–121.
    """
    factor = (K - N0) / N0
    growth = K / (1 + factor * np.exp(-mu * (t - lag)))
    return growth, lag + np.log(factor) / mu


# ---------------------------------------------------------------------------
# Gompertz
# ---------------------------------------------------------------------------

def gompertz_growth(t, N0, K, mu, lag):
    """
    Modified Gompertz growth model (Zwietering et al., 1990).

    Re-parameterisation of the Gompertz function in biologically meaningful
    terms. Growth evolves in log-population space as:

        ln(N(t) / N0) = A · exp(−exp((µ · e / A) · (lag − t) + 1))

    where  A = ln(K / N0)  is the total log-range of growth.

    Unlike the logistic, this model sets N ≈ N0 at t = 0 (not at t = lag);
    the 'lag' parameter is the x-intercept of the tangent drawn at the
    inflection point, i.e. the standard microbiological lag-time definition.

    The inflection point (maximum growth rate) is at:

        t_inf = lag + ln(K / N0) / (µ · e)

    Reference
    ---------
    Zwietering MH, Jongenburger I, Rombouts FM, van't Riet K (1990).
    Modeling of the bacterial growth curve.
    Applied and Environmental Microbiology, 56(6), 1875–1881.
    https://doi.org/10.1128/aem.56.6.1875-1881.1990
    """
    A = np.log(K / N0)                              # log-scale growth range
    exponent = (mu * np.e / A) * (lag - t) + 1
    growth = N0 * np.exp(A * np.exp(-np.exp(exponent)))
    t_inflection = lag + A / (mu * np.e)
    return growth, t_inflection


# ---------------------------------------------------------------------------
# Richards
# ---------------------------------------------------------------------------

def richards_growth(t, N0, K, mu, lag, v=1.0):
    """
    Richards (generalised logistic) growth model.

    Extends the logistic with a shape parameter v (ν > 0) that controls the
    skewness of the S-curve.  N(lag) = N0 for all values of v:

        N(t) = K / (1 + C · exp(−µ · v · (t − lag)))^(1/v)

    where  C = (K / N0)^v − 1.

    Shape behaviour of v
    --------------------
    v = 1         : symmetric logistic  (inflection at N = K/2)
    v → 0⁺        : Gompertz-like curve (inflection near N0)
    0 < v < 1     : inflection above K/2 (fast early growth, slow late)
    v > 1         : inflection below K/2 (slow early growth, fast late)

    The inflection point (N = K / (1 + v)^(1/v)) is at:

        t_inf = lag + ln(C / v) / (µ · v)

    For v = 1, C / v = (K − N0) / N0 and this reduces to the logistic
    inflection formula exactly.

    References
    ----------
    Richards FJ (1959). A flexible growth function for empirical use.
    Journal of Experimental Botany, 10(29), 290–301.
    https://doi.org/10.1093/jxb/10.2.290

    Turner ME, Bradley EL, Kirk KA, Pruitt KM (1976). A theory of growth.
    Mathematical Biosciences, 29(3–4), 367–373.
    https://doi.org/10.1016/0025-5564(76)90112-7
    """
    C = (K / N0) ** v - 1                           # shape scaling factor
    growth = K / (1 + C * np.exp(-mu * v * (t - lag))) ** (1 / v)
    t_inflection = lag + np.log(C / v) / (mu * v)
    return growth, t_inflection


# ---------------------------------------------------------------------------
# Baranyi–Roberts
# ---------------------------------------------------------------------------

def baranyi_growth(t, N0, K, mu, lag):
    """
    Baranyi–Roberts growth model (Baranyi & Roberts, 1994).

    A mechanistic model in which cells must first complete a physiological
    'work-to-do' before entering exponential growth.  An internal adjustment
    function A(t) acts as a physiological clock that transitions smoothly
    from 0 (during lag) to (t − lag) (post-lag):

        N(t) = N0 · K / (N0 + (K − N0) · exp(−µ · A(t)))

        A(t) = t + (1/µ) · ln((exp(−µ · t) + q0) / (1 + q0))

        q0   = 1 / (exp(µ · lag) − 1)     [initial physiological state]

    The shape of A(t) requires no extra parameter: it is fully determined
    by µ and lag.  As t → ∞, A(t) → t − lag and the model converges to
    the standard logistic.

    Inflection point
    ----------------
    The exact inflection satisfies  A(t_inf) = ln((K − N0) / N0) / µ,
    which has no closed form.  The asymptotic approximation (exact for
    t_inf ≫ 1/µ) is:

        t_inf ≈ lag + ln((K − N0) / N0) / µ

    Constraint: lag > 0  (lag = 0 gives q0 → ∞ and the model collapses
    to the logistic; use logistic_growth directly in that case).

    References
    ----------
    Baranyi J, Roberts TA (1994). A dynamic approach to predicting
    bacterial growth in food.
    International Journal of Food Microbiology, 23(3–4), 277–294.
    https://doi.org/10.1016/0168-1605(94)90157-0

    Baranyi J, Roberts TA (1995). Mathematics of predictive food
    microbiology.
    International Journal of Food Microbiology, 26(2), 199–218.
    https://doi.org/10.1016/0168-1605(94)00121-L
    """
    q0  = 1.0 / (np.exp(mu * lag) - 1)              # initial physiological state
    A_t = t + (1.0 / mu) * np.log((np.exp(-mu * t) + q0) / (1.0 + q0))
    growth = N0 * K / (N0 + (K - N0) * np.exp(-mu * A_t))
    t_inflection = lag + np.log((K - N0) / N0) / mu
    return growth, t_inflection