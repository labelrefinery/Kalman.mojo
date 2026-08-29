# Kalman.mojo

Kalman filtering and RTS smoothing for offline track refinement, in pure Mojo.

The state-estimation half of a tracking-by-detection auto-labeler. Pair it with
[Hungarian.mojo](https://github.com/labelrefinery/Hungarian.mojo) for the
data-association half.

## Why a smoother, not just a filter

An online filter is causal: the estimate at frame *k* may only use frames
0..*k*, because frame *k+1* has not happened yet. An offline auto-labeler has
the entire log on disk, so that constraint is self-imposed and expensive.

The Rauch-Tung-Striebel smoother runs the ordinary forward filter, then a
backward recursion that folds all future evidence into every past estimate.
That is the whole argument for offboard labeling, and it shows up three ways:

- **A detection at frame 90 sharpens the pose at frame 40.** Objects that were
  distant, occluded, or barely resolved early in a track inherit the accuracy
  they earn later, once they are close.
- **Gaps are interpolated by the dynamics, not by drawing a line.** An object
  occluded for 15 frames gets a physically consistent trajectory through the
  gap, with an honest covariance that widens in the middle.
- **Track starts stop being garbage.** A forward filter's first few estimates
  are dominated by the prior; the backward pass replaces that prior with
  evidence. Velocity at frame 0 is unobservable to a causal filter — it has
  seen exactly one position.

```mojo
from kalman import Observation, GaussianState, Matrix, smooth_track

var obs = List[Observation]()
obs.append(Observation(dt=0.1, z=[10.3]))
obs.append(Observation.gap(dt=0.1))       # occluded, no detection
obs.append(Observation(dt=0.1, z=[10.9]))

var prior = GaussianState([0.0, 0.0], Matrix.identity(2))
var track = smooth_track(
    prior, obs^, dims=1, accel_var=1.0, measurement_var=0.04
)
# track.states[k] is conditioned on the whole sequence, gaps included.
```

## API

`src/kalman/__init__.mojo` is the whole library. The types, in pipeline order:

| type / function | role |
| --- | --- |
| `Matrix` | dense row-major matrix: matmul, transpose, add/sub, Gauss-Jordan inverse |
| `GaussianState` | a belief: mean vector plus covariance |
| `constant_velocity(dims, dt)` | state transition, layout `[p…, v…]` |
| `white_noise_acceleration(dims, dt, accel_var)` | the classic piecewise-white-noise `Q` |
| `position_measurement(dims)` | observation matrix for a position-only detector |
| `predict(state, F, Q)` | propagate forward |
| `update(state, z, H, R)` | fold in a measurement; returns `UpdateResult` |
| `UpdateResult` | posterior plus `innovation` and `nis` |
| `Observation` | one frame; `Observation.gap(dt)` for a missed detection |
| `filter_forward(...)` | causal pass, recording the priors the smoother needs |
| `rts_smooth(priors, posteriors, transitions)` | the backward recursion |
| `smooth_track(...)` | forward filter + backward pass, one call per tracklet |
| `SmoothedTrack` | `states`, per-frame `nis`, `num_measurements` |

## Two numerical choices worth knowing about

**The covariance update is in Joseph form**, `P = (I-KH) P (I-KH)' + K R K'`.
More arithmetic than `(I-KH) P`, and worth it: Joseph stays symmetric positive
semi-definite under floating point error *and* under a suboptimal gain, which
is the regime you are in the moment anything about `R` is approximate. The
textbook form's failure mode is a covariance that silently loses symmetry over
a few thousand frames.

**Every update returns `nis`**, the normalized innovation squared `y' S⁻¹ y`.
It is chi-square distributed with `len(z)` degrees of freedom when the filter
is consistent, and it is what you should feed to an assignment solver's gate
rather than raw Euclidean distance — a fast, uncertain track deserves a wider
gate than a slow, well-observed one. Watch its running average too: a mean far
from `len(z)` means the noise model is wrong, not that the tracker is unlucky.

Pick `accel_var` from physics, not by tuning against a metric. For a tracked
excavator swinging its house, peak angular acceleration at the bucket is a
number you can measure once and reuse.

## Install

As a [mojoshelf](https://mojoshelf.org) tin, in pixi mode:

```
pixi shelf add kalman-mojo
```

or as a submodule:

```
shelf add kalman-mojo         # then: mojo run -I shelf/kalman-mojo/src ...
```

The registry and conda package name is `kalman-mojo`, while the Mojo import
name stays `kalman`:

```mojo
from kalman import smooth_track
```

## Tests

```
pixi run test
```

Fifteen cases. The matrix layer is checked by inverse round-trips on random
diagonally-dominant matrices; `nis` against a hand computation. The ones that
actually pin the smoother down:

- the last smoothed state equals the last filtered posterior (the RTS boundary
  condition)
- smoothing never increases a variance, at any frame — conditioning on more
  data cannot make an estimate less certain
- smoothed RMSE beats filtered RMSE on the same track, same noise, same model
- an occluded stretch is filled to within 0.35 m, with variance peaking in the
  middle of the gap and dropping at observed frames

## License

Apache 2.0.
