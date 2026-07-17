# asterodetect

`asterodetect` is an experimental Bayesian detector for broad power excesses
from solar-like oscillations.  Its first-stage probability model compares
three complete descriptions of a power-density spectrum:

1. frequency-independent noise;
2. noise plus Harvey-like granulation;
3. noise plus granulation and a Gaussian oscillation envelope.

The package currently contains the deterministic model, periodogram
likelihoods, whole-spectrum mixture calculation, simulations, JAX numerical
kernels, an AsteroScale sample adapter, and unit tests. Sampler integration
will be added after the core likelihood has been validated.

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
