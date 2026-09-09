---
layout: page
title: Contributing
description: How to participate in xaker development. Conventional commits, branch-less master, the coverage gate, the rubric gate, and the single-word naming rule.
permalink: /contributing/
---

**Audience: contributors — inside maintainers and outside pull-requesters.**

**Time: 5 minutes.**

This page documents the contributor workflow inside xaker. Every
section maps to a rule enforced in CI; the checklist at the bottom
is what every contribution needs to satisfy.

## Ground rules

- **Single-word naming.** Modules, classes, functions, methods,
  dataclass fields, and CLI flags are one word. No `_private`
  prefixes. No `import x as y`. No multi-word snake_case in
  algorithms. CI greps for this and the build fails on a hit.
- **Branch-less.** Commits land on `master` directly. Pull requests
  are optional for outside contributors; inside the maintainer
  workflow, fixes are pushed in focused, single-purpose commits.
- **Conventional commits.** Subject line of the form
  `<type>(<scope>): <subject>`. Types used in this repository:
  `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `ci`,
  `style`.
- **No fake citations.** CITATION.cff stays an honest
  "paper in preparation" until the arXiv identifier exists.
- **Coverage is enforced.** `--cov-fail-under=90` on
  every test run. A refactor that drops coverage below the gate
  fails the build.

## Workflow

1. **Branch.** `git checkout -b <type>/<short-slug>`. Example:
   `fix/pcg-cache-invalidation`.
2. **Edit.** Keep commits focused. One commit per issue or per
   change set. The body of a non-trivial commit cites the issue
   it closes (e.g. `Closes #37`).
3. **Test.** `pytest tests/ -v --cov=xaker --cov-fail-under=90
   --strict-markers -m "not slow"`. The full suite must remain
   green.
4. **Lint.** `pylint xaker/ --rcfile=pyproject.toml`. The `lint` job
   in CI also runs the single-word naming guard.
5. **Type-check.** `mypy xaker/ --ignore-missing-imports
   --no-implicit-optional --warn-unused-ignores`. New code without
   a `-> ReturnType:` annotation is auto-failed.
6. **Validate.** `xaker-validate --min-total 14`. The `rubric` job
   in CI runs this on every push to `master`.
7. **Build.** `python -m build`. The wheel must build cleanly;
   `twine check dist/*` should pass.
8. **Open a PR** (outside contributors) or **push** to `master`
   (inside).

## Adding a public symbol

Any name the user can `from xaker import X` requires:

- An entry in `xaker/__init__.py`'s `__all__`.
- A docstring following the [Google developer
  style](https://developers.google.com/style) on the public class
  or function, with sections: short summary line, then
  `Args:`, `Returns:` (where relevant), and a single-paragraph
  usage example.
- A test in `tests/test_<module>.py`. The test must assert shape,
  finiteness, and dispatch (for registries).
- An entry in `docs/api.md` if the symbol is reachable from a doc.

## Adding a benchmark

A new bench goes into `xaker/bench/<name>.py`. It must:

- Emit the schema-stable `Result` JSON via `xaker.bench.bench.write`.
- Be runnable from a YAML spec under `examples/specs/`.
- Commit at least one canonical `paper_runs/<name>.json` so the
  `repro` and `efficiency` rubric graders accept the change.
- Pass `xaker-validate` at ≥ 17/18 once the new JSON lands.

## Style review

Before sending a PR, run a style pass on the docs you touched.
Apply both style guides in full:

- The Archbee
  [technical-writing](https://www.archbee.com/blog/technical-writing-style-guide)
  four-step workflow (define audience, research, write, review)
  with one-paragraph orientations per doc, descriptive H1s, and
  scannable sections.
- The Google developer
  [style guide](https://developers.google.com/style): page
  titles with the page name first and site name last; semantic
  landmarks; present tense; second person where natural;
  inclusive language; no filler (`simply`, `just`, `easily`);
  descriptive link text; consistent terminology; tables with
  header rows; lists that start with a sentence fragment or verb.

## Reporting issues

Use [GitHub Issues](https://github.com/sachncs/xaker/issues). The
minimum reproduction is described in
[Troubleshooting](/xaker/troubleshooting/#filing-an-issue).

## Pre-submit checklist

Run this list mentally before every commit:

- [ ] Did `pytest tests/ --cov=xaker --cov-fail-under=90` pass?
- [ ] Did `pylint xaker/ --rcfile=pyproject.toml` pass?
- [ ] Did `mypy xaker/ --ignore-missing-imports` pass?
- [ ] Did `xaker-validate --min-total 14` pass?
- [ ] Are all new public symbols in `xaker.__all__`?
- [ ] Are all new public symbols listed in `docs/api.md`?
- [ ] Are all new docs reviewed against the style guide?
- [ ] Is `CHANGELOG.md` updated for any user-facing change?
- [ ] Is `STATUS.md` coverage number within two points of HEAD?

If any item is unchecked, fix it before pushing.



## Next steps

- [Installation](/xaker/installation/) — set up the dev
  environment.
- [Architecture](/xaker/architecture/) — the polymorphism
  registries and the block dispatch.
- [API reference](/xaker/api/) — the public surface that new
  code must follow.
- [Recipes](/xaker/recipes/) — concrete patterns for adding
  variants, kernels, and preconditioners.
