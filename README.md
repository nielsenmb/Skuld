# asterodetect

`asterodetect` is an experimental Bayesian detector for broad power excesses
from solar-like oscillations.  Its first-stage probability model compares
three complete descriptions of a power-density spectrum:

1. frequency-independent noise;
2. noise plus Harvey-like granulation;
3. noise plus granulation and a Gaussian oscillation envelope.

The package contains the deterministic model, periodogram likelihoods,
whole-spectrum and prior-predictive mixture calculations, simulations, JAX
numerical kernels, an AsteroScale sample adapter, and unit tests.

The validated public API remains NumPy based. Performance-sensitive kernels
in `asterodetect.jax_backend` are pure, traceable functions that can be used
with `jax.jit`, `jax.vmap`, and automatic differentiation when worthwhile.
AsteroScale, Baldr, and Mimir are installed from pinned Git commits so that
changes on their main branches cannot silently alter Skuld's inference,
probability transforms, or spectral estimates.

The detection spectrum is intended to be heavily and irreversibly binned
before inference. `AsteroScaleSamples.bin_spectrum` chooses a fixed width of
approximately one predicted large separation, subject to retaining at least
five bins across the envelope FWHM. Spectral models are averaged over the
stored bin boundaries rather than evaluated only at their centres. An
overdispersion factor reduces the nominal Gamma shape when unresolved modes
or the spectral window introduce more variance than independent smooth bins.

AsteroScale's `A_env` is a radial-mode RMS amplitude, not envelope power.
`ObservationModel` converts each joint AsteroScale sample to the Gaussian
parameterization used by the detector:

```text
P_env = V_tot A_env^2 (FWHM_env / Dnu) sqrt(pi / (4 ln 2)) (D eta)^2
```

Here `V_tot` is the summed relative mode power per radial order (3.04 by
default), `D` is the target-flux dilution, and `eta` is the finite-integration
time amplitude response evaluated at `numax`. Both observational corrections
are explicit and act quadratically on power. The visibility should eventually
be inferred or sensitivity-tested rather than treated as exact.

```python
from asterodetect import AsteroScaleSamples, ObservationModel

samples = AsteroScaleSamples.infer(stellar_measurements)
observed = samples.envelope_parameters(
    ObservationModel(integration_time_seconds=120.0, dilution=0.92)
)
# observed contains aligned integrated_power, numax, and sigma samples.
```

Kallinger et al. (2014) define `A_gran` as the combined bolometric RMS of two
super-Lorentzian components. Skuld splits its *variance* between them:

```text
a_low^2 + a_high^2 = (A_gran / C_bol)^2
```

The default split is equal, but `granulation_variance_fraction_low` exposes
the assumption. Cadence attenuation is applied frequency by frequency to the
Harvey profiles, rather than once at their characteristic frequencies.

## Probability model

For an unbinned periodogram, each power measurement is exponentially
distributed about the limit spectrum.  For a spectrum averaged over `s`
independent bins, the corresponding Gamma distribution is used:

```text
P_j ~ Gamma(shape=s_j, scale=S_j / s_j)
```

The mixture is a mixture of complete-spectrum likelihoods:

```text
p(D) = sum_k pi_k prod_j p(P_j | S_k(nu_j))
```

It is intentionally not a product of per-bin mixtures.  Consequently, its
responsibilities describe support for whole spectral models rather than the
fraction of frequency bins assigned to a component.

The three classes should be interpreted observationally. The granulation-only
model does **not** claim that a solar-like star has no oscillations. It is the
null hypothesis that the observed PSD does not require a visible oscillation
envelope after granulation has been accounted for. The binary detection
probability is therefore

```text
P(detection | D) = P(oscillation | D)
P(no detection | D) = P(noise | D) + P(granulation | D)
```

Dropping the granulation-only model changes the question. A granulation
spectrum would then be compared with pure noise and with a model that contains
both granulation and oscillations. The latter can win simply because it is the
only remaining model with granulation, producing an apparent oscillation
detection even when no oscillation envelope was injected.

## Marginalizing the model parameters

`PriorPredictiveMarginalizer` integrates the three complete models over
aligned AsteroScale rows. White noise can be fixed or supplied as one prior
draw per row:

```python
from asterodetect import ObservationModel, PriorPredictiveMarginalizer

binned = samples.bin_spectrum(spectrum)
result = PriorPredictiveMarginalizer().evaluate(
    binned,
    samples,
    white_noise=white_noise_draws,
    observation=ObservationModel(integration_time_seconds=120.0),
)

print(result.responsibilities)
print(result.diagnostics["oscillation"].effective_sample_size)
```

For each model the evidence estimator is `logmeanexp` of the likelihoods.
Diagnostics include the importance-weight effective sample size and a
delta-method standard error for the log evidence. A model probability should
not be treated as calibrated when its evidence is dominated by very few prior
draws; more samples or a better proposal distribution are then required.

## End-to-end detector

`Detector` joins the individual stages without making the binned data depend
on a sampled parameter:

```python
from asterodetect import Detector, ObservationModel

detector = Detector(
    draws=4096,
    observation=ObservationModel(
        integration_time_seconds=120.0,
        dilution=0.92,
    ),
)
result = detector.run(
    unbinned_spectrum,
    stellar_constraints={"Teff": (5772.0, 80.0), "parallax": (10.0, 0.1)},
    rng=42,
)

print(result.classification)
print(result.probabilities)
print(result.evaluation.diagnostics)
```

The detector runs AsteroScale (or accepts an existing
`AsteroScaleSamples` object), chooses and freezes the bin width, draws aligned
nuisance parameters, and evaluates all three marginal likelihoods.
`NuisancePrior` currently represents uncertainty in:

- the white-noise floor;
- oscillation-envelope amplitude;
- combined granulation amplitude;
- the variance split between the two Kallinger components;
- the effective Gamma overdispersion.

The automatic white-noise centre is the median of the highest-frequency 20%
of the supplied PSD. This is a convenient initial estimate, not a generally
valid background measurement. Supply `white_noise_centre` to `Detector.run`
when that region contains appreciable stellar power or instrumental
structure.

## Injection recovery

Calibration cases retain arbitrary metadata, making it possible to group
results later by cadence, duration, magnitude, evolutionary state, dilution,
or injected amplitude:

```python
from asterodetect import InjectionCase, evaluate_injections

cases = [
    InjectionCase(
        name="solar-noise-001",
        truth="noise",
        spectrum=noise_spectrum,
        stellar_constraints=asteroscale_samples,
        metadata={"cadence_seconds": 120.0, "amplitude_scale": 0.0},
    ),
    InjectionCase(
        name="solar-osc-001",
        truth="oscillation",
        spectrum=oscillation_spectrum,
        stellar_constraints=asteroscale_samples,
        metadata={"cadence_seconds": 120.0, "amplitude_scale": 1.0},
    ),
]
calibration = evaluate_injections(detector, cases, seed=123)
print(calibration.confusion_matrix)
print(calibration.accuracy)
print(calibration.multiclass_brier_score)
```

The injected parameters should deliberately differ from the inference-prior
centre. Otherwise the exercise tests numerical self-consistency rather than
detection calibration.

Larger experiments can be constructed reproducibly with
`build_injection_grid`. The simulation policy remains in a user-supplied
factory, while the grid helper records every coordinate and repeat in the case
metadata:

```python
from asterodetect import build_injection_grid

axes = {
    "white_noise": [0.1, 0.3, 1.0],
    "amplitude_scale": [0.0, 0.3, 1.0],
    "duration_days": [27.4, 365.0],
}
cases = build_injection_grid(axes, make_injection, repeats=20, seed=123)
calibration = evaluate_injections(detector, cases, seed=456)

by_amplitude = calibration.group_by("amplitude_scale")
weak = calibration.subset(amplitude_scale=0.3, duration_days=27.4)
```

Here `make_injection(name, parameters, rng)` must return an `InjectionCase`.
The independent generator supplied to each call prevents results from
depending on loop order. Detector settings such as `dnu_scale` should be
compared with separate detector runs over the same cases, because they change
the inference rather than the injected population.

`build_detection_study` is a convenience for completeness/false-positive
experiments. It generates noise and granulation cases once per grid
coordinate, plus oscillation cases at each requested amplitude. This avoids
silently over-weighting the negative classes by repeating them along an
amplitude axis that does not affect them:

```python
from asterodetect import build_detection_study

cases = build_detection_study(
    {
        "white_noise": [0.1, 0.3, 1.0],
        "duration_days": [27.4, 365.0],
    },
    factory,
    oscillation_amplitudes=[0.1, 0.3, 1.0],
    repeats=20,
    seed=123,
)
calibration = evaluate_injections(detector, cases, seed=456)
metrics = calibration.detection_metrics(threshold=0.5)
print(metrics.completeness, metrics.false_positive_rate)
```

The returned binary metrics treat oscillation as the positive class and both
noise and granulation as negative. They include completeness, false-positive
rate, precision, binary Brier score, and a reliability table. Use
`threshold_curve(calibration)` to inspect the trade-off between completeness
and false positives rather than choosing a threshold from the same evaluation
sample. Empty denominators (for example, completeness in a null-only subset)
are reported as `NaN`, not as zero.

A runnable demonstration is provided in `examples/calibration_study.py`:

```bash
uv run --extra test python examples/calibration_study.py \
    --repeats 3 --draws 256 --output calibration-summary.json
```

This example is deliberately small. A scientific calibration needs many more
independent realizations and should reserve a separate validation population
for selecting the final probability threshold.

For a larger multi-regime checkpoint, use
`examples/astrophysical_campaign.py`. It constructs separate correlated
AsteroScale sample clouds and appropriate absolute noise axes for dwarfs,
subgiants, and low-luminosity red giants:

```bash
uv run python examples/astrophysical_campaign.py \
    --profile checkpoint --repeats 2 --draws 512 \
    --output astrophysical-campaign.json
```

The checkpoint profile spans 27.4- and 90-day baselines, two white-noise
levels per regime, and oscillation amplitudes of 0.3 and 1.0 times the
scaling-relation prediction. The `standard` profile additionally varies
dilution and the injected granulation amplitude, and inserts a 0.5-amplitude
case to map the transition between weak and readily detectable oscillations.

Every exact grid cell is divided by stochastic realization before inference
into independent tuning and validation populations. On tuning cases, the
campaign compares the original single-lognormal amplitude prior with several
normal-plus-suppressed mixtures, and chooses an operating threshold under a
5% false-positive constraint for each candidate. The candidate with the
highest tuning completeness is selected, with Brier score and false-positive
rate breaking ties. Only that frozen prior and threshold are then evaluated
on validation cases.

The mixture changes the nuisance distribution of oscillation amplitude
relative to each intact AsteroScale row; it does not replace or decorrelate
the AsteroScale sample cloud. The JSON report includes every candidate's
tuning metrics, the selected hyperparameters, and validation breakdowns by
regime, duration, amplitude, dilution, and granulation offset:

```bash
uv run python examples/astrophysical_campaign.py \
    --profile standard --repeats 4 --draws 512 \
    --maximum-false-positive-rate 0.05 \
    --output astrophysical-campaign-standard.json
```

At least two repeats are required for the split; four or more gives a less
fragile estimate in each half. Prior and threshold selection are analogous to
choosing an instrument configuration on calibration data before opening the
sealed science sample: validation spectra are not evaluated until both
choices have been frozen.

In the initial 480-case standard study, the tuning population retained the
original single-lognormal prior. Mixture candidates increased oscillation
probabilities for some negative cases as well as suppressed oscillators, so
the false-positive constraint forced higher thresholds and reduced tuning
completeness. The default prior is therefore unchanged. This is a calibration
checkpoint rather than evidence that suppressed stars do not exist; a larger
population model may need suppression probability to depend on stellar or
observational properties instead of using one global mixture weight.

Use `build_regime_detection_study` to construct the same pattern with real
AsteroScale samples or different regime definitions. It accepts one injection
factory and one grid-axis mapping per regime because equal absolute
white-noise levels are generally not equally informative across the HR
diagram.

### Evidence and binning sensitivity

Before increasing the astrophysical grid, use `run_sensitivity_study` to check
whether the prior-predictive evidence is stable and whether the chosen
frequency binning changes the answer:

```python
from asterodetect import run_sensitivity_study

study = run_sensitivity_study(
    cases,
    draw_counts=[128, 512, 2048],
    dnu_scales=[0.5, 1.0, 2.0],
    estimators=["prior", "adaptive"],
    repeats=4,
    seed=123,
)
for summary in study.summaries():
    print(summary)
for comparison in study.estimator_comparisons():
    print(comparison)
```

The injected spectra are paired: exactly the same periodogram realization is
used for every inference configuration. Independent inference repeats then
measure Monte Carlo variation in the evidence calculation. Each summary
reports probability scatter, classification accuracy, the lowest median ESS
fraction among the three models, and the largest median log-evidence standard
error. A high model probability is not considered converged merely because it
is close to zero or one.

When both estimators are requested, `estimator_comparisons()` reports the
adaptive-to-prior ESS and evidence-error ratios for every draw-count/bin-width
combination. It also reports the adaptive-to-prior repeat-scatter ratio, the
mean absolute change in oscillation probability, and the change in
classification accuracy. This makes it possible to check that improved ESS
corresponds to stable scientific conclusions, rather than treating sampling
efficiency alone as sufficient.

The runnable `examples/sensitivity_study.py` writes the same diagnostics to
JSON. `notebooks/03_evidence_and_binning_sensitivity.ipynb` illustrates how to
inspect the results.

### Astrophysical injections

`AstrophysicalInjectionFactory` supplies a standard simulation policy. It
uses a stochastic Lorentzian mode comb rather than the detector's smooth
Gaussian envelope, so injection recovery tests unresolved mode structure
instead of merely reproducing the inference model:

```python
from asterodetect import AstrophysicalInjectionFactory, build_injection_grid

factory = AstrophysicalInjectionFactory(asteroscale_samples)
cases = build_injection_grid(
    {
        "truth": ["noise", "granulation", "oscillation"],
        "duration_days": [27.4, 365.0],
        "cadence_seconds": [120.0],
        "white_noise": [0.1, 1.0],
        "amplitude_scale": [0.1, 0.3, 1.0],
    },
    factory,
    repeats=20,
    seed=123,
)
```

The factory builds a regular Fourier grid, two normalized Kallinger
super-Lorentzians, and approximate `l=0,1,2,3` ridges beneath a Gaussian
power envelope. It applies dilution and the frequency-dependent integration
response before drawing an exponential periodogram realization.

### Gaps and observing windows

For model-misspecification tests, the factory can instead realize the
stochastic spectrum as a light curve, multiply it by an observing mask, and
transform it back to a PSD:

```python
continuous = factory(
    "continuous",
    {"truth": "oscillation", "window_profile": "continuous"},
    np.random.default_rng(12),
)
gapped = factory(
    "gapped",
    {"truth": "oscillation", "window_profile": "tess-like"},
    np.random.default_rng(12),
)
```

Using the same random seed makes this a paired comparison: both cases begin
with the same Fourier realization and differ only in the observing window.
The built-in TESS-like approximation combines longer periodic downlink gaps,
shorter interruptions, and a small fraction of randomly missing cadences.
Its parameters are configurable through `tess_like_observing_window`; it is
not intended to reproduce the quality flags of a particular observed target.
For controlled experiments, `tess_observing_window` can construct the
continuous, momentum-dump-only, downlink-only, combined, and
`random-loss-matched` profiles separately. The matched-random profile removes
exactly as many cadences as the combined profile but scatters them randomly.
It is therefore a control for asking whether a result follows total observing
time or coherent gap structure.

`observing_window_diagnostics` reports the duty cycle, number of gaps, longest
gap, and strongest non-zero spectral-window sidelobe. These are properties of
the mask, not detector performance metrics.

When no observing mask is passed to `Detector.run`, the detector continues to
use pristine model spectra with its independent-Gamma likelihood. The
campaigns in this section deliberately use that default path, so they measure
robustness to unmodelled leakage and inter-bin correlations. The later
target-specific window-aware path corrects the expected mean spectrum while
retaining the same Gamma approximation.

### Comparing model sets

Completed evidences can be recombined under alternative model priors without
rerunning the detector:

```python
two_model = result.evaluation.reweight(
    {"noise": 0.5, "oscillation": 0.5}
)
```

This is intended as a paired diagnostic, not as evidence that an omitted
model is physically absent. For a whole calibration experiment,
`reweight_calibration_models` applies the same operation to every recovery.
The runnable `examples/model_set_study.py` compares the original three-model
calculation with noise-versus-oscillation and
granulation-versus-oscillation alternatives on identical evidence estimates.

In the 480-spectrum standard window campaign, removing the granulation model
gave an apparent 100% true-positive rate but a 50% false-positive rate: all
granulation-only validation spectra were called detections. No tested
threshold satisfied the 5% false-positive constraint. The three-model
calculation is therefore retained.

## Tutorials

The notebooks follow the detector in implementation order:

1. `01_models_and_binning.ipynb` — resolved modes and fixed
   \(\Delta\nu\)-scale binning;
2. `02_injection_recovery.ipynb` — generating spectra and running recovery;
3. `03_evidence_and_binning_sensitivity.ipynb` — Monte Carlo convergence;
4. `04_from_spectrum_to_probability.ipynb` — comparing a measured spectrum
   with the three complete models and converting evidences into probabilities;
5. `05_true_positive_rate.ipynb` — tuning/validation splits, thresholds,
   true-positive rates, observing windows, and the model-removal diagnostic.

Install the plotting dependencies with `uv sync --extra tutorial`. The last
notebook reads the saved standard-campaign summary in `notebooks/data` so it
can reproduce the reported plots without rerunning the full adaptive
inference campaign.

Run the leakage-safe paired checkpoint with:

```bash
uv run python examples/window_campaign.py \
  --profile checkpoint \
  --repeats 2 \
  --draws 256 \
  --output window-campaign.json
```

The threshold is selected from uninterrupted tuning cases under the requested
false-positive constraint, then frozen before either uninterrupted or gapped
validation cases are summarized. The JSON report includes true-positive and
false-positive rates, paired oscillation-probability shifts, detections gained
or lost, mean duty cycles, and breakdowns by amplitude, stellar regime, and
observing duration.

In the initial 480-spectrum standard campaign, the TESS-like window retained
95.3% of cadences. At the threshold selected from uninterrupted tuning cases,
held-out true-positive rate fell from 48.6% for continuous observations to
38.9% for gapped observations, while both false-positive rates were 4.2%.
Full-amplitude recovery fell from 95.8% to 79.2%. Across 120 matched validation
pairs, 13 detections were lost and 6 were gained. These remain calibration
checkpoint statistics, with two validation realizations per exact grid cell,
but they show that high duty cycle alone does not make spectral-window effects
negligible.

### Focused full-amplitude window study

`examples/focused_window_study.py` removes the deliberately suppressed
oscillators from the comparison and freezes the previously selected
probability threshold at 0.45. It applies five windows to each of 96
full-amplitude stellar realizations while sharing both the injected Fourier
realization and inference seed:

```bash
uv run python examples/focused_window_study.py \
  --repeats 8 \
  --draws 256 \
  --threshold 0.45 \
  --skip-interval-sweep \
  --output focused-window-study.json
```

The 480-evaluation study found:

| Window | Duty cycle | Full-amplitude TPR | Mean oscillation probability |
| --- | ---: | ---: | ---: |
| Continuous | 100.0% | 94.8% | 0.939 |
| Momentum dumps only | 99.2% | 86.5% | 0.853 |
| Downlinks only | 96.6% | 94.8% | 0.928 |
| Random loss matched to TESS-like | 95.3% | 95.8% | 0.932 |
| Combined TESS-like | 95.3% | 88.5% | 0.873 |

The combined window produced nine lost and three gained detections relative
to continuous data (exact paired \(p=0.146\)). The momentum-dump-only profile
produced ten lost and two gained detections (\(p=0.039\)). That loss was
localized to the low-luminosity-RGB cases; dwarfs and subgiants were
essentially unchanged.

The simulated RGB star has \(\Delta\nu \simeq 9.0\,\mu\mathrm{Hz}\). A
perfectly periodic 2.5-day momentum-dump pattern creates a spectral-window
comb spaced by \(4.63\,\mu\mathrm{Hz}\), close to \(\Delta\nu/2\). A targeted
32-realization RGB sweep compared operationally relevant recurrence
intervals. When gap durations were scaled so that every profile retained
approximately the same 99.2% duty cycle, the result was:

| Momentum-dump interval | Gap duration | Full-amplitude TPR | Mean oscillation probability |
| --- | ---: | ---: | ---: |
| 2.5 days | 30 min | 71.9% | 0.675 |
| 4.0 days | 48 min | 90.6% | 0.810 |
| 6.75 days | 81 min | 96.9% | 0.887 |

The median power integrated across the envelope FWHM remained within about
1% of the continuous value. The failure therefore follows distortion of the
spectral shape, not simple removal of envelope power. It is an exploratory
subgroup result rather than a final mission-wide calibration: actual TESS
momentum-dump cadence varies by sector, and later operations place dumps
during contacts. See the official
[TESS observations page](https://tess.mit.edu/observations/) and
[Sector 2 data-release notes](https://tasoc.dk/docs/release_notes/tess_sector_02_drn02_v02.pdf).

The saved reports used for these numbers are
`notebooks/data/focused_window_study.json`,
`notebooks/data/momentum_interval_sweep.json`, and
`notebooks/data/momentum_interval_matched_duty.json`.
The two interval reports can be reproduced with:

```bash
uv run python examples/focused_window_study.py \
  --interval-sweep-only --repeats 8 --draws 256 \
  --seed 2321 --output momentum-interval-sweep.json

uv run python examples/focused_window_study.py \
  --interval-sweep-only --match-interval-duty-cycle \
  --repeats 8 --draws 256 --seed 2321 \
  --output momentum-interval-matched-duty.json
```

### Target-specific window-aware forward model

When a target's regular-cadence time mask is available, `Detector.run` can
pass every predicted complete spectrum through the corresponding spectral
window before evaluating the likelihood:

```python
window = tess_observing_window(
    90.0,
    120.0,
    profile="momentum-dumps",
)
result = detector.run(
    spectrum,
    stellar_constraints,
    observing_window=window,
)
```

`SpectralWindowOperator` performs the discrete two-sided convolution with
\(|W(\nu)|^2\), using the same duty-cycle normalization as the time-domain
simulator, then averages the result over the detector's fixed bins. Noise,
granulation, and oscillation predictions all receive the same treatment.
The ordinary detector path remains unchanged when no mask is supplied.

The likelihood is still the independent-bin Gamma approximation. Window
awareness therefore corrects the expected mean spectrum but does not claim
that Fourier bins remain independent after gaps.

`examples/window_aware_study.py` first checks full-amplitude, noisy
low-luminosity-RGB stars with the problematic 30-minute/2.5-day gap pattern.
In a 48-case study with 16 oscillators and 32 negative controls:

| Evaluation | TPR | FPR | Binary Brier score |
| --- | ---: | ---: | ---: |
| Continuous, current model | 100.0% | 9.4% | 0.0258 |
| Gapped, current model | 93.8% | 6.2% | 0.0514 |
| Gapped, window-aware model | 100.0% | 6.2% | 0.0247 |

The sample is deliberately small: one negative classification changes the
FPR by 3.1 percentage points. It is a focused implementation check, not a new
mission-wide threshold calibration. The important paired result is that
window awareness recovered the lost oscillator without adding false
positives relative to the same gapped spectra.

Replaying PR #10's harder 32-oscillator RGB population gives:

| Evaluation | Detections | TPR | Mean oscillation probability |
| --- | ---: | ---: | ---: |
| Continuous, current model | 31/32 | 96.9% | 0.916 |
| Gapped, current model | 23/32 | 71.9% | 0.675 |
| Gapped, window-aware model | 31/32 | 96.9% | 0.902 |

The saved reports are `notebooks/data/window_aware_study.json` and
`notebooks/data/window_aware_rgb_replay.json`. Reproduce them with:

```bash
uv run python examples/window_aware_study.py \
  --repeats 4 --draws 256 --seed 411 \
  --output window-aware-study.json

uv run python examples/window_aware_study.py \
  --oscillation-replay --repeats 8 --draws 256 --seed 2321 \
  --output window-aware-rgb-replay.json
```

### Real TESS validation factory

`examples/tess_validation.py` separates network-dependent preparation from
the expensive detector run. The shipped `examples/tess_targets.csv` is a
24-target smoke sample:

- 12 confirmed main-sequence/subgiant detections selected across the
  frequency range of the
  [TESS Luminaries catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A%2BA/701/A285);
- 12 confirmed RGB detections spanning brightness and frequency from
  [HD-TESS](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/164/135).

Published `numax` and `dnu` values are reference answers only. The manifest
loader forbids them, and other seismic quantities, from entering the
AsteroScale constraints. During preparation the script queries independent
TIC effective-temperature and radius estimates, downloads the preferred
available light curves through Mimir, converts each product to ppm, removes
cadences rejected by the quality mask and a 5-sigma clip, and caches:

- `tic-<id>.npz`: zero-filled ppm light curve, exact observing mask, cadence,
  and aperture dilution from `CROWDSAP`;
- `tic-<id>.constraints.json`: independent AsteroScale inputs.

Install the downloader dependencies and prepare either the whole smoke sample
or a small first batch:

```bash
uv sync --extra tess-validation

uv run python examples/tess_validation.py --limit 2 prepare
```

Preparation expects systematics-corrected products; it does not apply a
generic high-pass filter because a filter safe for dwarfs can remove a red
giant's oscillation envelope. Gaps longer than 50 days are shortened to one
cadence, following Hatt et al. (2023), while ordinary sector, downlink, and
momentum-dump gaps remain in the target-specific spectral window.
The run stage passes only the observed cadences to Mimir's `nifty-ls`
power-density estimator. The cached zero-filled grid is used only to retain
the exact observing mask for Skuld's window-aware forward model.
The smoke manifest starts with standard SPOC products. Lund et al. (2025)
needed custom apertures for some bright stars, so an individual miss should be
checked against the published/custom extraction before it is attributed to
the detector. A locally extracted light curve can replace the cached NPZ
without changing the run or summary stages.

Run locally with the frozen adaptive estimator and 0.45 threshold:

```bash
uv run python examples/tess_validation.py --limit 2 run \
  --draws 256 --fft-workers -1 --window-row-batch-size 8

uv run python examples/tess_validation.py --limit 2 summarize \
  --output tess-summary.json
```

Each successful target is written atomically to its own JSON file. Re-running
the command skips completed targets, so a long campaign can resume after an
interruption. Use `--tic TIC_ID` to isolate one target or `--overwrite` to
repeat completed work.

Only confirmed literature detections contribute to the reported real-data
TPR. A future list of reported non-detections is kept as a separate challenge
set: its flagged fraction is not called an FPR because lack of a published
detection does not prove that oscillations are absent.

Tutorial notebooks are in `notebooks/`. Install their dependencies with
`uv sync --extra tutorial` and follow the five-notebook sequence listed in
the Tutorials section above. The final two notebooks connect the component
and convergence tutorials to calibrated model probabilities and TPR.

## Adaptive importance sampling

`AdaptiveImportanceSampler` provides a defensive two-stage estimator for
cases where direct prior averaging has a very small evidence ESS. A pilot
sample fits a likelihood-weighted multivariate Student proposal, which is
mixed with the original prior:

\[
q(\theta)=\epsilon p(\theta)+(1-\epsilon)t_\nu(\theta).
\]

When the ordinary pilot weights collapse onto too few points, the likelihood
is tempered only for fitting the proposal until a requested pilot ESS is
retained. The final evidence weights always use the full, untempered
likelihood, so this stabilization does not change the target evidence. The
Student proposal also has heavier tails than the previous Gaussian fit.

The evidence estimate uses the complete importance correction
\(p(D\mid\theta)p(\theta)/q(\theta)\). The prior component preserves support
if the pilot misses a mode, and the result reports the final ESS and
log-evidence standard error, together with the pilot ESS and adaptation
temperature.

```python
from asterodetect import AdaptiveImportanceSampler

sampler = AdaptiveImportanceSampler(
    pilot_draws=256,
    draws=2048,
    defensive_fraction=0.2,
    pilot_ess_fraction=0.1,
    proposal_degrees_of_freedom=5,
)
result = sampler.run(prior_sample, prior_logpdf, log_likelihood, rng=42)
print(result.log_evidence, result.diagnostic)
```

The callables use an `(n_draws, n_parameters)` unconstrained parameter
matrix. `Detector` can now use this estimator directly:

```python
detector = Detector(
    estimator="adaptive",
    draws=2048,
    pilot_draws=256,
    defensive_fraction=0.2,
)
result = detector.run(spectrum, asteroscale_samples, rng=42)
```

AsteroScale remains the stellar prior: complete sample rows are resampled
without fitting a Gaussian to them, preserving correlations between
`numax`, `dnu`, amplitudes, and timescales. The adaptive proposal targets
only standardized nuisance coordinates, separately for each spectral model.
Each nuisance point averages the likelihood over several intact AsteroScale
rows (`stellar_draws_per_nuisance=8` by default). This reduces noise in the
proposal fit without factorizing or Gaussianizing the empirical stellar
sample cloud.
The original estimator remains the default while calibration is ongoing.

Paired comparisons can reuse each injected PSD across both estimators:

```python
study = run_sensitivity_study(
    cases,
    estimators=("prior", "adaptive"),
    draw_counts=(128, 512, 2048),
    dnu_scales=(0.5, 1.0, 2.0),
    repeats=4,
    seed=42,
)
for comparison in study.estimator_comparisons():
    print(comparison)
```

Adaptive sampling should not be made the default solely because an individual
case has a larger ESS. The paired study should show consistently higher ESS,
lower log-evidence error, and stable probabilities across inference seeds and
binning choices. `minimum_truth_model_median_ess_fraction` reports convergence
for each injection's known generating model; the original worst-model metric
is retained because even strongly disfavoured evidences can still be
diagnostically useful.

## Development install

```bash
python -m pip install -e ".[test]"
pytest
```

## Minimal example

```python
import numpy as np

from asterodetect import (
    GaussianEnvelope,
    HarveyComponent,
    PowerSpectrum,
    SpectralModel,
    WholeSpectrumMixture,
    simulate_periodogram,
)

frequency = np.linspace(100.0, 4000.0, 20_000)
granulation = HarveyComponent(power=2.0, characteristic_frequency=700.0)
envelope = GaussianEnvelope(
    integrated_power=150.0,
    numax=1800.0,
    sigma=250.0,
)

oscillation_model = SpectralModel(
    white_noise=1.0,
    harvey_components=(granulation,),
    envelope=envelope,
)

power = simulate_periodogram(frequency, oscillation_model, rng=42)
spectrum = PowerSpectrum(frequency, power)

mixture = WholeSpectrumMixture(
    models={
        "noise": SpectralModel(white_noise=1.0),
        "granulation": SpectralModel(
            white_noise=1.0,
            harvey_components=(granulation,),
        ),
        "oscillation": oscillation_model,
    },
    probabilities={"noise": 1 / 3, "granulation": 1 / 3, "oscillation": 1 / 3},
)

evaluation = mixture.evaluate(spectrum)
print(evaluation.responsibilities)
```
