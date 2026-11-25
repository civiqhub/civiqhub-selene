---
orphan: true
---

# Formula test page

This page is in no table of contents. It shows each kind of formula the site writes, so that a change to KaTeX, to its configuration or to the theme can be checked at a glance, in light and dark modes, on a wide and a narrow screen. The formulas of the first sections describe no method of the engine; the last section writes the definition that the docstring of a response model gives. Its letters are those of [Notation](notation.md).

## Inline

A formula inside a sentence: the levels run from $1$ to $K$, and the state $s_{ij}$ of a shown cell is one of them, or $0$ when the cell carries no vote. Greek letters: $\theta$, $\phi$, $\alpha$, $\beta$, $\tau$.

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
\left(\mu_{ql}\right) =
\begin{pmatrix}
\mu_{11} & \mu_{12} & \cdots & \mu_{1L} \\
\vdots & \vdots & \ddots & \vdots \\
\mu_{Q1} & \mu_{Q2} & \cdots & \mu_{QL}
\end{pmatrix}
\in \mathbb{R}^{Q \times L}
$$

## Nested subscripts and superscripts

$$
a_{i_{1} j_{2}}^{(n+1)} = b^{g^{h}} + e_{k_{m}}^{2}
$$

## Operators and fonts

$$
\hat{u} = \operatorname*{arg\,max}_{u \in \mathbb{R}} \; \mathbb{E}\left[\log g(\boldsymbol{y} \mid u)\right]
$$

## A long formula

On a narrow screen this formula scrolls on its own line; the page does not widen.

$$
h(x) = \sum_{n=0}^{20} \frac{x^{n}}{n!} - \prod_{m=1}^{12} \left(1 + \frac{x}{m^{2}}\right) + \int_{0}^{x} \frac{\sin u}{u} \, du + \log\left(1 + e^{x}\right) - \frac{1}{2} \log\left(1 + x^{2}\right) + \sqrt{1 + x^{2} + x^{4} + x^{6} + x^{8} + x^{10} + x^{12} + x^{14} + x^{16} + x^{18} + x^{20}}
$$

## In a table

| Quantity | Formula |
| --- | --- |
| Number of levels | $K$ |
| A probability | $p_k \in [0, 1]$ |
| A sum | $\sum_{k=1}^{K} p_k = 1$ |

## In an admonition

:::{note}
A formula inside a note: $\Phi(z) = \frac{1}{2} \left(1 + \operatorname{erf}\frac{z}{\sqrt{2}}\right)$.

$$
\Phi(-z) = 1 - \Phi(z)
$$
:::

## A formula of the code: the level probability of the ordinal probit

{py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` reads a vote as a hidden normal opinion. In the co-cluster $(q, l)$, the opinion has the mean $\mu_{ql}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_mean`) and the standard deviation $\sigma_{ql}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.latent_spread`), and fixed thresholds $t_1 < \dots < t_{K-1}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.thresholds`) cut it into $K$ levels, with $t_0 = -\infty$ and $t_K = +\infty$. A shown cell carries a vote with the probability $\nu_{ql}$ ({py:attr}`~civiqhub_selene.blocks.responses.OrdinalProbitParameters.vote_rate`). The probability $F_{qls}$ of each state $s$ is then:

$$
\begin{aligned}
F_{ql0} &= 1 - \nu_{ql}, \\
F_{qlk} &= \nu_{ql} \, p_{qlk}, \\
p_{qlk} &= \Phi\!\left(\frac{t_k - \mu_{ql}}{\sigma_{ql}}\right) - \Phi\!\left(\frac{t_{k-1} - \mu_{ql}}{\sigma_{ql}}\right),
\end{aligned}
$$ (eq-test-ordinal-probit)

where $\Phi$ is the standard normal distribution function. By default the thresholds are $t_k = k + \tfrac{1}{2}$ for $k = 1, \dots, K - 1$, so that $\mu_{ql}$ reads on the axis of levels. Equation {eq}`eq-test-ordinal-probit` is the definition that the docstring of {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit` writes in ASCII.
