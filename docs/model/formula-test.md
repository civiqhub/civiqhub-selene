---
orphan: true
---

# Formula test page

This page is in no table of contents. It shows each kind of formula the site writes, so that a change to KaTeX, to its configuration or to the theme can be checked at a glance, in light and dark modes, on a wide and a narrow screen. The formulas of the first sections describe no method of the engine; the last section writes the definition that the docstring of a response model gives.

## Inline

A formula inside a sentence: the levels run from $1$ to $K$, and a vote $x_{ij}$ of the voter $i$ on the item $j$ is one of them, or $0$ when there is no vote. Greek letters: $\theta$, $\phi$, $\alpha$, $\beta$, $\tau$.

## Numbered, with a reference

$$
\sum_{k=1}^{K} p_k = 1, \qquad p_k \geq 0
$$ (eq-test-sum)

Equation {eq}`eq-test-sum` is numbered by Sphinx, and this sentence refers to it.

## Aligned equations (amsmath)

The environment `align` stands on its own lines, without `$$`. Sphinx gives it one number; KaTeX's own numbers are hidden.

\begin{align}
a &= b + c \\
d &= e^{f} - g_{h}
\end{align}

## Cases

$$
\operatorname{sign}(x) =
\begin{cases}
-1 & \text{if } x < 0, \\
0 & \text{if } x = 0, \\
1 & \text{if } x > 0.
\end{cases}
$$

## Matrix

$$
\boldsymbol{\theta} =
\begin{pmatrix}
\theta_{11} & \theta_{12} & \cdots & \theta_{1L} \\
\vdots & \vdots & \ddots & \vdots \\
\theta_{Q1} & \theta_{Q2} & \cdots & \theta_{QL}
\end{pmatrix}
\in \mathbb{R}^{Q \times L}
$$

## Nested subscripts and superscripts

$$
x_{i_{1}j_{2}}^{(t+1)} = \alpha^{\beta^{\gamma}} + \tau_{k_{\ell}}^{2}
$$

## Operators and fonts

$$
\hat{\phi} = \operatorname*{arg\,max}_{\phi} \; \mathbb{E}\left[\log p(\boldsymbol{x} \mid \phi)\right]
$$

## A long formula

On a narrow screen this formula scrolls on its own line; the page does not widen.

$$
\log p(\boldsymbol{x}) = \sum_{i=1}^{n} \sum_{j=1}^{d} \sum_{q=1}^{Q} \sum_{\ell=1}^{L} \sum_{k=1}^{K} \mathbb{1}\{x_{ij} = k\} \, z_{iq} \, w_{j\ell} \, \log \left( \frac{\exp(\theta_{q\ell k})}{\sum_{h=1}^{K} \exp(\theta_{q\ell h})} \right) + \sum_{i=1}^{n} \sum_{q=1}^{Q} z_{iq} \log \alpha_q + \sum_{j=1}^{d} \sum_{\ell=1}^{L} w_{j\ell} \log \beta_\ell
$$

## In a table

| Quantity | Formula |
| --- | --- |
| Number of levels | $K$ |
| A probability | $p_k \in [0, 1]$ |
| A sum | $\sum_{k=1}^{K} p_k = 1$ |

## In an admonition

:::{note}
A formula inside a note: $\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-u^2/2} \, du$.

$$
\Phi(-z) = 1 - \Phi(z)
$$
:::

## A formula of the code: the level probability of the ordinal probit

{py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` reads a vote as a hidden normal opinion, with mean $\mu$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_mean`) and standard deviation $\sigma$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_spread`), cut into $K$ levels by fixed thresholds $t_1 < \dots < t_{K-1}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.thresholds`), with $t_0 = -\infty$ and $t_K = +\infty$. A shown cell carries a vote with the probability $\rho$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.vote_rate`):

$$
\begin{aligned}
P(\text{no vote}) &= 1 - \rho, \\
P(\text{vote at level } k) &= \rho \, p_k, \\
p_k &= \Phi\!\left(\frac{t_k - \mu}{\sigma}\right) - \Phi\!\left(\frac{t_{k-1} - \mu}{\sigma}\right),
\end{aligned}
$$ (eq-test-ordinal-probit)

where $\Phi$ is the standard normal distribution function. By default the thresholds are $t_k = k + \tfrac{1}{2}$ for $k = 1, \dots, K - 1$, so that $\mu$ reads on the axis of levels. Equation {eq}`eq-test-ordinal-probit` is the definition that the docstring of {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` writes in ASCII.
