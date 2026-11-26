# KaTeX

This folder holds a copy of [KaTeX](https://katex.org), the library that renders the formulas of the documentation site in the browser. The site loads it from here and from no other server.

The files are copied from the published package as they are: none is modified.

## Provenance

| | |
| --- | --- |
| Version | 0.19.1, the latest stable version on 25/11/2026 (published on 23/10/2026) |
| Archive | <https://registry.npmjs.org/katex/-/katex-0.19.1.tgz>, the npm package published by the KaTeX project |
| SHA-256 of the archive | `24da5a599eb2fcede2e270b99a6ff48bca816beecc95feeb84e8cccbd1bb34fd` |
| Integrity of the archive (npm, sha512) | `sha512-jG8uKntjSEcm2EpQOh0XturpPuuC9r3Sq5HLHdsFdWZT27E672SF19QEi+of0/YeOy15lDgRhKvr5614uRrEfA==` |
| Source | <https://github.com/KaTeX/KaTeX>, commit `8000fbd4e29cca2c0cb1e88b745af3ed2f436b66` (the `gitHead` the npm registry gives for 0.19.1) |

The release archive of the GitHub repository was not reachable from the environment where the copy was made. The npm package holds the same `dist/` folder, built by the KaTeX project.

## Files

| File of this folder | File of the archive | SHA-256 |
| --- | --- | --- |
| `katex.min.css` | `package/dist/katex.min.css` | `00e0b094171b07f58422075f469b4feac2b2f7bce1aeec943c08da4d697f4dcf` |
| `katex.min.js` | `package/dist/katex.min.js` | `fa140498b8cfc8302cbe5e7cb1bdbc1f952bc38b3fa258ede09e73edbf9a74bc` |
| `contrib/auto-render.min.js` | `package/dist/contrib/auto-render.min.js` | `e5372d199bcdae8b4de71d0f7ceba72a4ba12774a27c60a6f1f77d03b3228ee4` |
| `fonts/*.woff2` (20 files) | `package/dist/fonts/*.woff2` | Listed by the command below |
| `LICENSE` | `package/LICENSE` | `766ccc1f306c885aa45542a9846bbd0a505b27a0374f146778171c2254ce18e3` |
| `fonts/OFL.txt` | Not in the archive: written for this copy | |

Left out: the fonts in `.woff` and `.ttf` (current browsers all read `woff2`), the files that are not minified, `katex-swap*.css`, `katex.mjs`, and the other scripts of `contrib/`. The style sheet still names the `.woff` and `.ttf` files after the `woff2` ones; a browser that reads `woff2` never asks for them.

## Licenses

- The code (`katex.min.css`, `katex.min.js`, `contrib/auto-render.min.js`): MIT License, Copyright (c) 2013-2020 Khan Academy and other contributors. The text is in `LICENSE`.
- The fonts (`fonts/*.woff2`): SIL Open Font License 1.1, Copyright (c) 2009-2010 Design Science, Inc. and Copyright (c) 2014-2018 Khan Academy, with the Reserved Font Names `KaTeX_*`. The archive gives this license in the `name` table of each font, and ships no text of it: `fonts/OFL.txt` holds these notices and the text of the license. The fonts are not modified, so the Reserved Font Names do not restrict this copy.

`NOTICE`, at the root of the repository, cites both.

## Check the copy

```bash
curl -sSfLO https://registry.npmjs.org/katex/-/katex-0.19.1.tgz
echo "24da5a599eb2fcede2e270b99a6ff48bca816beecc95feeb84e8cccbd1bb34fd  katex-0.19.1.tgz" | sha256sum -c
tar xzf katex-0.19.1.tgz
cmp package/LICENSE docs/_static/katex/LICENSE
cmp package/dist/katex.min.css docs/_static/katex/katex.min.css
cmp package/dist/katex.min.js docs/_static/katex/katex.min.js
cmp package/dist/contrib/auto-render.min.js docs/_static/katex/contrib/auto-render.min.js
for font in package/dist/fonts/*.woff2; do
  cmp "$font" "docs/_static/katex/fonts/$(basename "$font")"
done
```

Every `cmp` prints nothing when the copy is identical to the archive.

## Update KaTeX

1. Download the archive of the new version from the npm registry, or the release archive from GitHub. Check an npm archive against the sha512 integrity the registry publishes, and record its SHA-256.
2. Replace the files of this folder by those of the archive, with the same selection as above.
3. Read the `name` table of the fonts again: if the copyright or the license changed, update `fonts/OFL.txt` and `NOTICE`.
4. Update this file and the paragraph KaTeX of `NOTICE`: version, address, SHA-256.
5. Build the site and check that the formulas still render.
