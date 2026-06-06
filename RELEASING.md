# Releasing

How a release gets cut. The heavy lifting lives in
`.github/workflows/release.yml`; this file is the human-side checklist.

## 0. One-time PyPI setup (not done yet)

Publishing uses OIDC trusted publishing, no API token. Before the first PyPI
upload someone has to register the publisher on pypi.org:

1. Log in to pypi.org, go to "Publishing" -> "Add a new pending publisher".
2. Project name `opticore`, owner `qorexdevs`, repo `opticore`,
   workflow `release.yml`, environment `pypi`.
3. Same on test.pypi.org with environment `testpypi` if you want dry-runs.
4. Flip the gate: `gh variable set PYPI_READY --body true`.

Until `PYPI_READY` is set, tag pushes still build wheels and publish a GitHub
Release, they just skip the PyPI upload.

## 1. Cutting a release

1. Make sure main is green and the working tree is clean.
2. Decide the version from the Unreleased section of `CHANGELOG.md`
   (any breaking change bumps minor while we're pre-1.0).
3. Move Unreleased to a dated `## [X.Y.Z]` section, bump `version` in
   `pyproject.toml`, commit, push, wait for CI.
4. Tag and push:

   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. The Release workflow builds wheels (linux/macos/windows x cp310-cp313),
   an sdist, and creates the GitHub Release with everything attached.
   Takes ~20-30 min because of the cibuildwheel matrix.
6. Edit the generated release notes: lead with breaking changes and their
   migration lines, then the highlights from the changelog.

## 2. TestPyPI dry-run (optional)

Run the workflow manually with `publish_to_testpypi: true`:

```
gh workflow run release.yml -f publish_to_testpypi=true
```

Needs the testpypi pending publisher from step 0.

## Wheel matrix notes

- No musllinux/Alpine and no 32-bit: no demand so far, cibuildwheel `skip`
  covers it. Revisit if someone asks.
- manylinux2014 floor: works on every glibc distro from the last ~10 years
  and has prebuilt numpy/pandas wheels.
- Every wheel is smoke-tested post-build (pytest minus plot/benchmark tests).
