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
