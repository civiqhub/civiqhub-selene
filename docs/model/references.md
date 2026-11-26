# References

Checked against the code of `develop` at commit `97a0f15`.

## In plain words

These are the works that the code of the engine or a page of this section cites, and no other. Each entry says which names of the code cite it, in their docstring or in the {py:attr}`~civiqhub_selene.estimator.Estimator.reference` of the method, and which pages of this section do. A citation says where a model or an algorithm comes from; the formulas of this section are those the code computes, checked against the code.

## Works

### Airoldi, Blei, Fienberg and Xing (2008)

Airoldi, E. M., Blei, D. M., Fienberg, S. E. and Xing, E. P. (2008). Mixed membership stochastic blockmodels. *Journal of Machine Learning Research*, 9, 1981–2014. [jmlr.org/papers/v9/airoldi08a.html](https://www.jmlr.org/papers/v9/airoldi08a.html)

- Code: {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`, docstring and reference, for the mixed membership.
- Pages: [The mixed-membership model](mixed-membership.md).

### Biernacki, Celeux and Govaert (2000)

Biernacki, C., Celeux, G. and Govaert, G. (2000). Assessing a mixture model for clustering with the integrated completed likelihood. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(7), 719–725. [doi:10.1109/34.865189](https://doi.org/10.1109/34.865189)

- Code: {py:class}`~civiqhub_selene.blocks.LatentBlockModel`, docstring, for the integrated completed likelihood.
- Pages: [Choosing the shape](choosing-the-shape.md).

### Corneli, Bouveyron and Latouche (2020)

Corneli, M., Bouveyron, C. and Latouche, P. (2020). Co-clustering of ordinal data via latent continuous random variables and not missing at random entries. *Journal of Computational and Graphical Statistics*, 29(4), 771–785. [doi:10.1080/10618600.2020.1739533](https://doi.org/10.1080/10618600.2020.1739533)

- Code: {py:class}`~civiqhub_selene.blocks.responses.OrdinalProbit`, docstring; {py:class}`~civiqhub_selene.blocks.LatentBlockModel` and {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`, docstring and reference, for the ordinal response model.
- Pages: [The response models](response-models.md), [The latent block model](latent-block-model.md), [The mixed-membership model](mixed-membership.md).

### Dempster, Laird and Rubin (1977)

Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1–22. [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)

- Code: {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`, docstring and reference, for expectation-maximisation.
- Pages: [The response models](response-models.md), [The mixed-membership model](mixed-membership.md).

### Dhillon (2001)

Dhillon, I. S. (2001). The code cites the author and the year only, for the Laplacian without regulariser.

- Code: {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding`, docstring.
- Pages: [The starts](initialization.md).

### Govaert and Nadif (2003)

Govaert, G. and Nadif, M. (2003). Clustering with block mixture models. *Pattern Recognition*, 36(2), 463–473. [doi:10.1016/S0031-3203(02)00074-2](https://doi.org/10.1016/S0031-3203(02)00074-2)

- Code: {py:class}`~civiqhub_selene.blocks.LatentBlockModel`, docstring and reference, for the latent block model.
- Pages: [The latent block model](latent-block-model.md).

### Pfeffermann (1993)

Pfeffermann, D. (1993). The code cites the author and the year only, for the weighted pseudo-likelihood, with Skinner (1989).

- Code: {py:class}`~civiqhub_selene.blocks.LatentBlockModel` and {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`, docstring.
- Pages: [The latent block model](latent-block-model.md), [The mixed-membership model](mixed-membership.md).

### Pratt (1981)

Pratt, J. W. (1981). Concavity of the log likelihood. *Journal of the American Statistical Association*, 76(373), 103–106. [doi:10.1080/01621459.1981.10477613](https://doi.org/10.1080/01621459.1981.10477613)

- Code: `fit_thresholds` of `civiqhub_selene.blocks._numerics`, docstring, for the concavity of the likelihood of an ordinal probit in its thresholds, which makes the Newton steps on the estimated thresholds safe.
- Pages: [The response models](response-models.md#estimated-thresholds).

### Qin and Rohe (2013)

Qin, T. and Rohe, K. (2013). Regularized spectral clustering under the degree-corrected stochastic blockmodel. *Advances in Neural Information Processing Systems*, 26, 3120–3128. [arXiv:1309.4111](https://arxiv.org/abs/1309.4111)

- Code: {py:class}`~civiqhub_selene.blocks.SpectralStart`, {py:class}`~civiqhub_selene.blocks.spectral.SpectralEmbedding`, {py:meth}`~civiqhub_selene.blocks.spectral.SpectralEmbedding.labels` and {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding`, docstrings, for the regulariser, the projection onto the sphere (Lemma 3.3) and thresholded spectral clustering.
- Pages: [The starts](initialization.md).

### Rohe, Qin and Yu (2016)

Rohe, K., Qin, T. and Yu, B. (2016). Co-clustering directed graphs to discover asymmetries and directional communities. *Proceedings of the National Academy of Sciences*, 113(45), 12679–12684. [doi:10.1073/pnas.1525793113](https://doi.org/10.1073/pnas.1525793113)

- Code: {py:class}`~civiqhub_selene.blocks.SpectralStart` and {py:func}`~civiqhub_selene.blocks.spectral.spectral_embedding`, docstrings, for DI-SIM and its regularised Laplacian.
- Pages: [The starts](initialization.md).

### Rubin (1976)

Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592. [doi:10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581)

- Code: none.
- Pages: [The data](data.md), for the cells that are not shown, missing at random.

### Skinner (1989)

Skinner, C. J. (1989). The code cites the author and the year only, for the weighted pseudo-likelihood, with Pfeffermann (1993).

- Code: {py:class}`~civiqhub_selene.blocks.LatentBlockModel` and {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel`, docstring.
- Pages: [The latent block model](latent-block-model.md), [The mixed-membership model](mixed-membership.md).

## Other sources the code names

The docstring of {py:class}`~civiqhub_selene.blocks.MixedMembershipBlockModel` names the CiviqHub explainer "LBM and MM-OLBM step by step" as the reference engine that the role per block and the exact expectation over the profiles follow.
