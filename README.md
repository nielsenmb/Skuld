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
AsteroScale is installed from a pinned Git commit so that changes on its main
branch cannot silently alter Skuld's inference behaviour.

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

Every exact grid cell is divided by stochastic realization into independent
tuning and validation populations. The tuning population chooses the most
complete probability threshold whose false-positive rate is no larger than
5% by default. That threshold is then applied unchanged to the validation
population. The JSON report retains the fixed 0.5-threshold results and adds
the selected threshold plus validation breakdowns by regime, duration,
amplitude, dilution, and granulation offset:

```bash
uv run python examples/astrophysical_campaign.py \
    --profile standard --repeats 4 --draws 512 \
    --maximum-false-positive-rate 0.05 \
    --output astrophysical-campaign-standard.json
```

At least two repeats are required for the split; four or more gives a less
fragile estimate in each half. Threshold selection is analogous to choosing
an instrument setting on calibration data before opening the sealed science
sample: validation cases must not influence the operating point.

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
response before drawing an exponential periodogram realization. Regular
sampling is intentional in this first version; gaps and spectral windows
belong in a later model-misspecification experiment.

Tutorial notebooks are in `notebooks/`. Install their dependencies with
`uv sync --extra tutorial`, then start with
`01_models_and_binning.ipynb` and continue to
`02_injection_recovery.ipynb`, then
`03_evidence_and_binning_sensitivity.ipynb`.

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
