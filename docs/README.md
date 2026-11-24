# Documentation

This folder holds the pages of the documentation site of `civiqhub-selene`. The site is built with [Sphinx](https://www.sphinx-doc.org) from two sources: the Markdown pages of this folder, and the docstrings of the code.

This file explains how the site is organised and how to change it. It is read on GitHub and is not a page of the site.

## Build the site

The build needs Python 3.13 or later.

```bash
python -m pip install -e ".[viz]" --group docs
python -m sphinx -W -b html docs docs/_build/html
```

The reference imports `civiqhub_selene.viz`: install the `viz` extra, which includes `blocks`. Then open `docs/_build/html/index.html`. The option `-W` turns every warning into an error, as in the CI.

This builds the documentation of the code you have, as one site without versions. The published site holds several versions: see [How the site is published](#how-the-site-is-published).

## What is where

| Path | Content |
| --- | --- |
| `conf.py` | The configuration of Sphinx |
| `_ext/selene_docs.py` | A local Sphinx extension; its docstring says what it does |
| `_templates/sidebar/version-switcher.html`, `_static/version-switcher.js`, `_static/version-switcher.css` | The version selector of the sidebar |
| `_templates/sidebar/brand.html`, `_templates/sidebar/search.html`, `_templates/page.html` | The mark of the site, the search field, and the theme-migration script |
| `_static/civiqhub.css`, `_static/theme-migrate.js`, `_static/sidebar-wrap.js`, `_static/search-shortcut.js` | The CiviqHub theme (light and dark), the one-time theme migration, wrapping of sidebar module names, and the Ctrl K shortcut |
| `_static/fonts/` | Bricolage Grotesque (OFL 1.1) and `OFL.txt` |
| `_static/civiqhub-logo-light.svg`, `_static/civiqhub-logo-dark.svg`, `_static/civiqhub-symbol-light.svg`, `_static/civiqhub-symbol-dark.svg` | The CiviqHub wordmark and symbol, light and dark |
| `_static/logo.svg`, `_static/favicon.ico` | Former glyph and ICO, unused until NOTICE is updated |
| `index.md` | The home page and the table of contents of the site |
| `getting-started.md`, `library-map.md`, `architecture.md`, `examples.md` | The guide: pages written by hand. `examples.md` presents the notebooks of `examples/`, which live outside the site |
| `reference/` | The reference: one page per public module, generated from the docstrings |
| `changelog.md` | Includes `CHANGELOG.md` from the root of the repository |
| `_build/` | The output of the build. Not committed |

Sections that are planned and not written yet: the methods (definition, parameters and limits of each method), what the results do not say, and how to write an estimator outside this repository.

## Thème du site

Le site publié propose les modes clair, sombre et auto (préférence du système). Il reprend le style de la documentation produit CiviqHub sur GitBook, et le thème fusionné de `civiqhub-py` (commit `9365fe7`) : marque en tête de barre latérale, recherche en pilule, légendes de section en capitales, colonne de lecture, cartes d'accueil, avec le bleu `#2C448B` comme fil conducteur. L'adresse publique du site est <https://selene.civiqhub.com/>.

### Police Bricolage Grotesque

Trois graisses statiques, sous-ensemble latin (accents français, guillemets, ponctuation de code), licence SIL OFL 1.1.

| Fichier | Graisse | sha256 |
| --- | --- | --- |
| `docs/_static/fonts/BricolageGrotesque-Regular.woff2` | 400 | `e621e8a37e58558ca0e1dab28e2989854a79d89926879377788c400329a68bde` |
| `docs/_static/fonts/BricolageGrotesque-SemiBold.woff2` | 600 | `31302f47b289491baa9154e8ba49107173ca3461db2e62382e5d9b785fadaefd` |
| `docs/_static/fonts/BricolageGrotesque-Bold.woff2` | 700 | `522c92d99315b6c4b4fc9efb11857c318cc0e64f955d5a28707601cbbf5b4344` |

`docs/_static/fonts/OFL.txt` est la licence OFL 1.1 livrée à côté.

Provenance : le dépôt [`google/fonts`](https://github.com/google/fonts), dossier `ofl/bricolagegrotesque`, commit `06bfe0d62a98375cb09c942fa18e68bd32cff057`. Source variable : `BricolageGrotesque[opsz,wdth,wght].ttf`. Reconstruction (instances `opsz=14`, `wdth=100`, `wght` 400 / 600 / 700, puis sous-ensemble latin Google Fonts compressé en `woff2`) :

```bash
git clone --filter=blob:none --sparse https://github.com/google/fonts.git
cd fonts
git sparse-checkout set ofl/bricolagegrotesque
git checkout 06bfe0d62a98375cb09c942fa18e68bd32cff057

LATIN="U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"
SRC="ofl/bricolagegrotesque/BricolageGrotesque[opsz,wdth,wght].ttf"

for spec in Regular:400 SemiBold:600 Bold:700; do
  name="${spec%:*}"
  weight="${spec#*:}"
  fonttools varLib.instancer "$SRC" opsz=14 wdth=100 "wght=${weight}" \
    --output "BricolageGrotesque-${name}.ttf"
  pyftsubset "BricolageGrotesque-${name}.ttf" \
    --unicodes="$LATIN" \
    --layout-features="*" \
    --flavor=woff2 \
    --output-file="BricolageGrotesque-${name}.woff2"
done
cp ofl/bricolagegrotesque/OFL.txt OFL.txt
```

`civiqhub.css` déclare trois `@font-face` (`font-display: swap`). Texte courant en 400, titres H2 à H4 et légendes de barre latérale en 600, H1 en 700. La police de code reste celle de Furo.

### Fichiers du thème

| Fichier | Rôle |
| --- | --- |
| `_static/civiqhub.css` | Tokens clair et sombre, `@font-face`, barre latérale, recherche, cartes, colonne de lecture, mermaid |
| `_static/theme-migrate.js` | Une fois, retire `localStorage.theme = "light"` laissé par l'ancien verrou |
| `_static/sidebar-wrap.js` | Enveloppe les noms de modules aux points dans la barre latérale |
| `_ext/civiqhub_carbon.py` | Style Pygments sombre, palette carbone seulement |
| `_static/search-shortcut.js` | Ctrl K / Cmd K focusent le champ de recherche |
| `_static/civiqhub-logo-light.svg` | Logo complet CiviqHub, mode clair (`#1f1f1e`) |
| `_static/civiqhub-logo-dark.svg` | Logo complet CiviqHub, mode sombre (`#fff`) |
| `_static/civiqhub-symbol-light.svg` | Symbole seul, favicon par défaut |
| `_static/civiqhub-symbol-dark.svg` | Symbole seul, favicon si `prefers-color-scheme: dark` |
| `_static/logo.svg` | Ancien glyphe, encore dans le dépôt, non utilisé |
| `_static/favicon.ico` | Ancien favicon ICO, encore dans le dépôt, non utilisé |
| `_static/home.svg` | Glyphe de maison du H1 d'accueil |
| `_static/github.svg` | Icône GitHub du pied de page |
| `_static/fonts/` | Les trois `woff2` et `OFL.txt` |
| `_static/screenshots/` | Captures de relecture du thème. Le build les retire de `_static/` du site publié |
| `_static/version-switcher.js`, `_static/version-switcher.css` | Sélecteur de version de la publication, natif à ce dépôt, au style du thème |
| `_templates/page.html` | Charge `theme-migrate.js` dans le `<head>` ; favicon sombre ; fil d'Ariane ; enveloppe d'accueil |
| `_templates/sidebar/brand.html` | Marque : logo complet clair et sombre |
| `_templates/sidebar/search.html` | Pilule : une loupe, « Rechercher... », badge Ctrl K |
| `_templates/sidebar/version-switcher.html` | Emplacement du sélecteur de version, propre à ce dépôt |

Les fichiers du thème repris de `civiqhub-py` portent l'en-tête Apache-2.0 de ce dépôt. `NOTICE` cite leur origine, Bricolage Grotesque, le glyphe, et le pied de page adapté de Furo (MIT).

### Tokens

Les couleurs et arrondis clairs viennent de `civiqhub-web` au SHA `22756d33636bb8b341e7c41ec8846c421867dc10`, fichier `packages/civiqhub-ui/src/tokens/generated.css`. Ce sont des valeurs de bootstrap : aucun contrat du design system n'est `ready`. Les valeurs sombres sont les neutres carbone du design system (Décision Yanis du 22/11/2026, 14:44 heure de Paris) : aucune teinte bleue.

| Variable CSS | Clair | Sombre | Rôle |
| --- | --- | --- | --- |
| `--civiqhub-primary` | `#2C448B` | `#FAFAFA` | Remplissage de marque en clair ; texte de marque en sombre |
| `--civiqhub-primary-foreground` | `#ffffff` | `#171717` | Texte sur la marque |
| `--civiqhub-background` | `#ffffff` | `#171717` | Fond de page (web `neutral-900` en sombre) |
| `--civiqhub-shell-surface` | `#FAFAFA` | `#000000` | Barre latérale, cartes, encarts, admonitions, annonce |
| `--civiqhub-muted` | `#f1f5f9` | `#404040` | Entrée active, survol, kbd, bordures UI |
| `--civiqhub-border` | `#e2e8f0` | `#404040` | Bordures UI |
| `--civiqhub-foreground` | `#020617` | `#FAFAFA` | Texte principal, liens |
| `--civiqhub-muted-foreground` | `#64748b` | `#e5e5e5` | Texte secondaire (contenu). En sombre, les menus de la barre sont `#FAFAFA` |
| `color-brand-primary` / `color-brand-content` | `#2C448B` | `#FAFAFA` | Liens et accents. En sombre, soulignement `#868686` puis `#FAFAFA` au survol |
| `--civiqhub-success` | `#047857` | `#73b876` | Avertissements tip / hint |
| `--civiqhub-destructive` | `#dc2626` | `#ff2f1c` | Avertissements danger / error |
| warning / caution | Furo | `#F2A202` | Titres d'admonition warning et caution |
| note / seealso | Furo | `#868686` | Titres d'admonition note ; jamais comme texte sur `#404040` |
| `--civiqhub-radius-sm` / `md` / `lg` / `xl` | 4 / 8 / 12 / 16 px | identiques | Arrondis |
| `--civiqhub-card-banner-from` / `to` | `color-mix` de primary 28 % puis 10 % avec le fond | `#171717` / `#000000` | Bandeaux des cartes |
| `--civiqhub-announcement` | primary 10 % dans le fond | `#000000` | Bandeau de la branche de développement |
| blocs de code | `#f1f5f9` | `#000000` | Fond des blocs. Jetons : texte `#FAFAFA`, commentaires `#868686`, chaînes `#73b876`, nombres `#F2A202`, erreurs `#ff2f1c`, noms `#e5e5e5` |

Le mode clair ne change pas. Le sombre n'utilise que les couleurs du tableau, plus des transparences de ces valeurs ou de `#000000`. `#868686` n'est jamais posé comme texte sur `#404040` (contraste 2,85:1). Police mono garde Furo, en attendant [#650](https://github.com/civiqhub/civiqhub-web/issues/650). Le favicon est le symbole CiviqHub fourni par Yanis le 22/11/2026 (clair par défaut, sombre si `prefers-color-scheme: dark`). Liens clairs en `primary` `#2C448B` : provisoire jusqu'à l'arbitrage du ds:gap [#648](https://github.com/civiqhub/civiqhub-web/issues/648).

Les SVG `civiqhub-logo-*.svg` et `civiqhub-symbol-*.svg` sont des actifs de marque CiviqHub fournis par Yanis le 22/11/2026. `logo.svg` et `favicon.ico` restent dans le dépôt, non utilisés, jusqu'à la PR qui mettra `NOTICE` à jour.

Écarts volontaires par rapport à `civiqhub-py` :

- les bandeaux des cartes mélangent le primary à 28 % puis 10 % (py : 18 % puis 6 %), plus proches du lavis du GitBook ;
- le bandeau d'annonce de la branche de développement est un lavis de primary à 10 %, texte foreground, et peut aller à la ligne (Furo le pose en barre noire, `nowrap`) ;
- les noms de modules dans l'arbre de référence s'enveloppent aux points (`<wbr>` après chaque `.`) ;
- la colonne de navigation fait 18 em, pour le libellé complet du sélecteur de version et les noms de sous-modules ;
- les tableaux du guide défilent à l'horizontale sous 48 em, pour rester lisibles à 390 px.

### Modes clair, sombre et auto

Furo 2025.12.19 lit `localStorage.getItem("theme")` dans `base.html` et pose `body[data-theme]` (`light`, `dark` ou `auto`). Le bouton de bascule natif est visible (clair, sombre, auto). Le mode par défaut est `auto` : la préférence `prefers-color-scheme` du système.

`theme-migrate.js` s'exécute dans le `<head>` (`_templates/page.html`) **et** via `html_js_files`. Si la clé `theme` vaut `"light"` et que `civiqhub-theme-migrated` est absente, il supprime `theme` puis pose la marque. Les visiteurs passés depuis la fusion de la PR n°10 avaient `theme = "light"` écrit par `force-light.js`, pas par un choix. Après cette migration, Furo retombe sur `auto`. Un visiteur qui choisit ensuite le mode clair garde `theme = "light"` : la marque empêche de l'effacer à nouveau.

`civiqhub.css` redéfinit les variables `--civiqhub-*` sous `body[data-theme="dark"]` et, pour `auto`, sous `@media (prefers-color-scheme: dark)` pour `body:not([data-theme="light"])`, comme Furo. `html { color-scheme }` suit le mode. `docs/conf.py` définit `dark_css_variables` (neutres carbone) et `pygments_dark_style = "civiqhub_carbon.CiviqHubCarbonDark"`.

Les diagrammes mermaid sont colorés avec les tokens du thème. `sphinxcontrib-mermaid` 2.1.1 observe `body[data-theme]` et relance le rendu sans rechargement.

### Extensions et dépendances

Groupe `docs` de `pyproject.toml` : `sphinx==9.2.0` (BSD-2-Clause), `myst-parser==5.2.0` (MIT), `furo==2025.12.19` (MIT), `sphinxcontrib-mermaid==2.1.1` (BSD-2-Clause), `sphinxext-opengraph==0.13.0` (BSD-3-Clause). `sphinxext-opengraph` pose `ogp_site_url` d'après `html_baseurl` ; les cartes sociales PNG sont désactivées (`ogp_social_cards.enable = False`) parce que le build n'a pas de réseau. Matplotlib est installé par l'extra `viz`, que la référence importe ; il ne sert pas aux cartes. L'origine publique est `https://selene.civiqhub.com/`. Si le script de publication passe encore `https://civiqhub.github.io/civiqhub-selene/`, `conf.py` la réécrit, chemin de version compris. Extension locale : `docs/_ext/selene_docs.py`, inchangée.

### Build local

```bash
python -m pip install -e ".[viz]" --group docs
python -m sphinx -W -b html docs docs/_build/html
python .github/scripts/build-docs-site.py --output _site --development HEAD
```

La première commande construit un site sans versions. La seconde reproduit la publication (`dev/` pour HEAD). Python 3.13 ou plus récent.

## How the site is published

The workflow `.github/workflows/docs.yml` publishes the site to GitHub Pages, at <https://civiqhub.github.io/civiqhub-selene/>. It runs when `develop` or `main` moves, and it can be started by hand: Actions, workflow `docs`, "Run workflow". It never runs on a pull request.

The published site holds one folder per version:

| Folder | Built from | Shown in the selector as |
| --- | --- | --- |
| `X.Y.Z/` | `main`, for the version `main` holds | `X.Y.Z (latest release)` |
| `X.Y.Z/` | The tag `vX.Y.Z`, for each earlier release | `X.Y.Z` |
| `dev/` | `develop` | `dev (development branch)` |

The address of the site redirects to the latest release, or to `dev` as long as `main` holds no documentation.

Every publication rebuilds every version, each from its own commit and with the tools this commit pins: GitHub Pages replaces the whole site each time. The script is `.github/scripts/build-docs-site.py`. Run it to build the site as it is published, from the references of your clone:

```bash
git fetch origin develop main --tags
python .github/scripts/build-docs-site.py --output _site
python -m http.server --directory _site
```

The version selector of the sidebar reads `versions.json`, a file the script writes next to the folders. It only appears on the published site, or behind a web server as above: a page opened from the disk has no selector. The `documentation` job of the CI runs the same script on the commit of the pull request alone, and keeps the result as the artifact `docs-html`.

A release needs nothing more for its documentation: when `main` receives the release, the site gains its folder. To change the address of the site, change `SITE_URL` in the workflow and the `Documentation` address in `pyproject.toml`.

## Five rules

1. **Pages are Markdown.** They are read by [MyST](https://myst-parser.readthedocs.io), and they must stay readable on GitHub.
2. **The reference is never written by hand.** To change what the reference says, change the docstring.
3. **Every public module has a page in the reference.** `tests/unit/test_docs_reference.py` checks it.
4. **Every Python block runs.** `tests/unit/test_docs_examples.py` runs the Python blocks of `README.md` and of every page of this folder, and compares what they print with the output the page shows.
5. **Every reference to a name of the engine resolves.** The build fails on a name that does not exist.

## Add a page to the guide

1. Create `docs/my-page.md`. It starts with one title of level 1.
2. Add `my-page` to a `toctree` of `index.md`. A page that is in no table of contents fails the build.
3. Build the site.

Link to another page with a relative Markdown link: `[Architecture](architecture.md)`. Link to a heading with the anchor GitHub gives it: `[the contract](architecture.md#the-contract-of-a-method)`.

## Refer to a name of the engine

Use a role, so that the name becomes a link to the reference and the build checks that it exists:

````markdown
{py:class}`~civiqhub_selene.blocks.VoteMatrix` holds the votes.
Call {py:meth}`~civiqhub_selene.estimator.Estimator.estimate`.
````

| Role | For |
| --- | --- |
| `{py:class}` | A class or an error |
| `{py:func}` | A function |
| `{py:meth}` | A method |
| `{py:attr}` | An attribute or a property |
| `{py:data}` | A constant |
| `{py:mod}` | A module |
| `{py:obj}` | Any of them |

The leading `~` shows the last part of the name only. Both import paths work: the package that exports the name (`civiqhub_selene.blocks.VoteMatrix`) and the module that defines it (`civiqhub_selene.blocks.data.VoteMatrix`).

On GitHub a role shows as plain text. This is accepted.

## Write a Python block

````markdown
```python
from civiqhub_selene.ordinal import VoteCounts

print(VoteCounts.from_scores([2, 2, -1]).total)
```

```text
3
```
````

- A block fenced with `python` is run by the tests. The blocks of a page share one namespace and run in the order of the page.
- A `text` block that directly follows a Python block is its output: the test compares it with what the block prints, character for character. Show an output only when it does not depend on the machine or on the version of a dependency.
- To show code that cannot run, such as a skeleton, write the comment `<!-- not run: the reason -->` on the line before the block, followed by an empty line.
- A page whose blocks use NumPy or `civiqhub_selene.blocks` is run in the test jobs that install NumPy only; a page whose blocks use `civiqhub_selene.viz` also needs NetworkX and Matplotlib. Nothing has to be declared.
- Examples use synthetic data only: `make_votes`, or values written by hand.
- `ruff format` also formats the Python blocks of the Markdown files.

## Write formulas and diagrams

A formula is written between dollar signs, inline (`$a + b$`) or on its own lines between `$$`. A diagram is a block fenced with `mermaid`; GitHub draws it as well.

The site loads MathJax and Mermaid from the jsDelivr network, on the pages that hold a formula or a diagram only.

## The reference

Each public module has one page, `reference/<dotted name of the module>.md`. The page of a module is three lines:

````markdown
# `civiqhub_selene.blocks.data`

```{eval-rst}
.. automodule:: civiqhub_selene.blocks.data
```
````

It shows the docstring of the module, then every name of its `__all__` with its docstring. The page of a package shows the docstring of the package, the names the package exports (the directive `public-names` reads its `__all__`) and the list of its modules.

When you add a public module:

1. Declare its `__all__`.
2. Create its page: copy the page of a sibling module and change the name on both lines.
3. Add the dotted name of the module to the `toctree` of the page of its package, for example `reference/civiqhub_selene.blocks.md`.

When you add a name to an existing module, add it to `__all__`: the page follows. When you remove or rename a module, remove or rename its page.

A module whose name starts with an underscore is internal and has no page. An underscore method appears in the reference when it carries its own docstring: this is how a hook such as `Estimator._compute` is documented. An override without a docstring does not appear.

## When the build fails

| Message | Cause | What to do |
| --- | --- | --- |
| `py:class reference target not found: civiqhub_selene...` | A docstring or a page names something that does not exist, or that is not public | Fix the name. A name is public when it is in the `__all__` of a public module |
| `py:class reference target not found: decimal.Decimal` (a name of another package) | A signature uses a type of a package that `conf.py` does not know yet | Add the top-level package to the first pattern of `nitpick_ignore_regex` in `conf.py` |
| `py:class reference target not found: Some words` | A `Returns:` section starts with `some words: more words`; the words before the colon are read as a type | Reword the first line of the section without a colon |
| `document isn't included in any toctree` | A page is in no table of contents | Add it to a `toctree` |
| `cross-reference target not found` (`myst.xref_missing`) | A Markdown link points to a page or a heading that does not exist | Fix the link |
| `Unexpected indentation`, `Block quote ends without a blank line` | A docstring is not valid reStructuredText | Put an empty line before and after a list or an indented block; write `::` before a literal block |

Sphinx reads no other site during a build, and the site does not link to the documentation of Python or NumPy.
