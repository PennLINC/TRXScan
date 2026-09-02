# Response to review round 10

**The blocker is real and it was mine.** Round 9's coil-sensitivity cleanup replaced a sub-voxel
registration error with a half-FOV one, and the test I wrote for it was structurally incapable of
noticing. The review's diagnosis is correct in every particular, including the mechanism, the
magnitude, why the test missed it, and the recommended fix. Both requested changes are done, plus
the PR-description refresh.

Commit `bd7a87f`. Rust: 95 tests under `--features cli`, 88 under default features, 74 Python
tests. The 24-set fixture grid regenerates byte-identical and `ACCEPTANCE: PASS` with an unchanged
table.

---

## 1. The coil-sensitivity coordinate origin — confirmed, fixed

Confirmed exactly as described. `build_coil_kspace` was passing

```rust
(x as f64 - sxs as f64 - xoff) / ox as f64        // sxs = snx / 2
```

into `coil_sensitivity`, whose coil ring is centred on `nx / 2` and whose other caller — the
Roemer combine — walks `x in 0..nx`. That expression is centred on the image and spans
`[-nx/2, nx/2)`. So the forward model's sensitivity field sat half a FOV from the combine's.

The review is right that this is worse than what it replaced in two ways. It is a half-FOV error
rather than a quarter-voxel one, and unlike the original it is present at `o = 1`, where `xoff = 0`
and the whole displacement is the `-sxs` term. Measured end to end: a 6-coil Roemer combine
differed from the single-coil image by **28% of peak at `o = 1`**.

What I got wrong conceptually is worth naming, because it is not a typo. I wrote a comment
asserting that coil sensitivities and eddy polynomials "is expressed in these coordinates" — as if
"acquired-voxel coordinates" were one thing. It is two things: a *unit* (acquired voxels rather
than sim cells) and an *origin* (grid corner rather than image centre). The eddy polynomial is an
expansion about the image centre and needs the centred origin; `coil_sensitivity` is written in
absolute grid coordinates because that is what the combine can supply. I unified on the unit and
silently unified the origin along with it.

Fixed as recommended — two coordinates, named apart, with the distinction spelled out rather than
implied:

```rust
//   xa — ABSOLUTE on the acquired grid, 0 .. nx-1 ... This is the frame the Roemer
//        combine walks (`x in 0..nx`), so anything the combine must agree with —
//        `coil_sensitivity`, whose coil ring is centred on nx/2 — uses it.
//   xc — CENTRED on the acquired image, -nx/2 .. nx/2-1. The eddy polynomial is
//        an expansion about the image centre and needs this one.
let (xa, ya) = ((x as f64 - xoff) / ox as f64, (y as f64 - yoff) / oy as f64);
f_real *= acq.signal_scale * coil_sensitivity(coil, ncoils, xa, ya, nx, ny);
```

I took the explicit-naming form over the `xc + xs` shorthand for the reason given: the shorthand
re-derives one frame from the other at the point of use, which is how the two got confused in the
first place. The eddy block keeps its own `(xc, yc, zc)` and is now **byte-identical to its
pre-round-9 form**, so eddy behaviour is unchanged rather than merely believed unchanged.

## 2. Why the round-9 test didn't catch it — the more important finding

The review is right, and this is the part I want to be explicit about rather than fold into the
fix. The round-9 test computed its own simulation-cell coordinate:

```rust
let xc = (v * o + i) as f64 / o as f64 - xoff / o as f64;   // == (v*o + i - xoff) / o
```

That is the *correct absolute* coordinate. So the test was pinning the arithmetic of the transform
the forward model **ought** to use, and comparing it against the combine — while the forward model
used a different one entirely. A test that supplies its own coordinate to the function under test
cannot possibly detect the production caller supplying a different coordinate. It was, structurally,
a test of my intent rather than of the code.

That is a general lesson and not a one-off: **a unit test of a helper cannot validate a call site.**
I had built a careful non-vacuity argument for that test — reconstructing the old convention inline
and asserting it was 146× worse — which made it look rigorous while it was measuring the wrong
thing. Rigour about the wrong quantity reads exactly like rigour.

Replaced at the level that matters, following the review's recommendation directly:

```
multicoil_roemer_reproduces_the_single_coil_image_at_every_oversampling
```

Full Fourier, no noise, no distortion, no relaxation, an off-grid box object, `n_coils = 6` versus
`n_coils = 1`, at `o = 1, 2, 4`. It goes through `simulate_slice` → `build_coil_kspace` → the
Roemer combine, so it exercises the production coordinate transform rather than restating it.

| o | registered (fixed) | unregistered (round 9) |
|---|---|---|
| 1 | 0.0 exactly | 0.283 |
| 2 | 1.08e-3 | — |
| 4 | 1.00e-3 | — |

`o = 1` is exact because the forward and combine grids coincide, so the combine divides out the
sensitivities identically. At `o > 1` the residual is not zero and should not be: the combine is
exact only for sensitivities band-limited within the acquired band, and these Gaussians are very
smooth but not exactly band-limited. Hence a 2% tolerance against a ~1e-3 measurement, which still
leaves two orders of magnitude of headroom to the 0.28 failure. Confirmed it fails on the round-9
code before I applied the fix.

The helper test is kept — the sub-voxel arithmetic is still worth pinning — but its doc comment now
opens by stating its scope and naming the end-to-end test as the one that covers the call site, so
it cannot be mistaken for coverage it does not provide again.

## Blast radius

Stating this precisely rather than reassuringly. The regression was introduced in `24d81c6` and
fixed in `bd7a87f`, both on this branch; it never appeared on `main`.

Where it does **not** reach:

- The Gibbs benchmark forces `n_coils = 1`, and `coil_sensitivity` returns `1.0` before touching
  the coordinate. Regenerated all 24 fixture sets after the fix and diffed against the round-9
  run: **byte-identical**. The acceptance table is identical too, down to the last digit.
- `trxscan --coils` defaults to `1`, so a default production run was unaffected as well.

Where it does: any run with `--coils > 1`, at any oversampling factor, and the GRAPPA path with it.
That is the production multi-coil simulation, which is exactly how the review characterised it.

## 3. The shear check didn't check for shear — confirmed, fixed

Also correct, and the reasoning is exactly right: `zoom_i` *is* the norm of source column *i*, so
scaling that column by `VOX / zoom_i` forces the result's norm to `VOX` regardless of the angles
between columns. Comparing target column norms to `VOX` is therefore tautological with respect to
shear. Verified rather than assumed — a source affine with a 0.447 direction-cosine off-diagonal
produces target column norms of exactly `(1.7, 1.7, 1.7)` and sailed through.

So the claim in my last response ("a sheared source affine is rejected explicitly") was false. The
guard did something useful — it catches a units regression, and it is what fired when I reverted
`step` to `diag([VOX] * 3)` to prove the test was non-vacuous — but that is not what I said it did.

Now shear is rejected on the **source**, using the review's suggested Gram-matrix test:

```python
directions = A[:3, :3] / zooms
gram = directions.T @ directions
if not np.allclose(gram, np.eye(3), atol=1e-6):
    raise SystemExit(f"source affine is sheared (worst off-diagonal {off:.4g} ...)")
```

This accepts arbitrary rotation and obliquity — an orthonormal basis in any orientation — and
rejects only genuine shear, where "VOX mm isotropic" would describe a parallelepiped rather than a
cube. The norm check stays, relabelled as what it actually is.

Two tests, in both directions, because a rejection check that rejects too much is as wrong as one
that rejects nothing: a 23° rotated 1 mm source is **accepted** and its output is 1.7 mm isotropic;
a source with the 0.447 off-diagonal is **rejected** with `sheared` in stderr. Confirmed the
rejection test fails when the guard is disabled.

## 4. PR description refresh

Done, and version-controlled this time at `docs/superpowers/pr-description.md` so it stops drifting
from the branch. Changes:

- Test counts: 83 Rust / 68 Python → **95 (`cli`) / 88 (default) / 74 Python**.
- CI wording: the body said CI "runs both suites plus `cargo check`". It now describes the three
  configurations CI actually *tests*, and says plainly why that gap mattered — `default = []` and
  the `io` module gated behind a feature the default build does not enable, so the geometry tests
  pinning the benchmark affine ran nowhere.
- Dropped the line-ending caveat, which `.gitattributes` settled.
- Added a caveat that the `microstructure` DIPY oracle is a **skip, not a pass**, so nobody reads
  the test count as covering it.
- Review history: seven rounds → ten, with round 9 and round 10 called out as the two worth
  reading, round 10 described as what it was rather than glossed.

**I have not applied it to the PR or pushed anything.** Three commits sit on the branch locally.
Applying the body is `gh pr edit 4 --body-file docs/superpowers/pr-description.md`.

---

## Verification

```
cargo test                                   88 passed
cargo test --features cli --all-targets      95 passed
cargo test --features cli,par --all-targets  95 passed
python -m pytest scripts -q                  74 passed
clippy --features cli --all-targets          45 warnings, unchanged
trxscan-benchmark matrix=64 oversample=4     24 sets, byte-identical to round 9
acceptance.py <24-set grid>                  PASS, table identical to round 9
```

## On the review

Two things worth saying back.

The first is that the review's structural point is the durable one. It would have been easy to
report this as "wrong sign on an offset" and move on. The actual finding is that I validated a
coordinate transform by re-deriving it in the test — and that this is the *third* coordinate-frame
error on this branch, after the object half-cell registration and the coil scale. The first two
were caught by tests that ran the production path (the analytic-oracle profile test, and the
end-to-end ringing test). This one was not, because I wrote the test at helper level. The pattern
is not "I keep getting offsets wrong", it is "helper-level tests keep failing to cover call sites",
and that is actionable in a way the first framing is not.

The second is narrower. The review notes the blocker "exists only because the agent chose to repair
a previously non-blocking multi-coil issue in the same round." That is a fair account of the cause,
and it is a reasonable argument for having left it alone. I would still make the same call, but for
a weaker reason than I gave last round: I said "the fix is smaller than the comment explaining why
it was left", and that was true of the code and false of the risk. The better justification is that
the code was already wrong and would have stayed wrong, and the round surfaced a genuine defect in
how I was testing it — which was worth more than the fix. But that is a benefit realised only
because the change was reviewed, not a reason the change was safe to make unreviewed.
