"""Kalman filtering and RTS smoothing for offline track refinement, in Mojo.

The state-estimation half of a tracking-by-detection auto-labeler. Pair it with
a linear assignment solver for the data-association half.

The reason this library exists separately from any online tracker is the
*smoother*. An online filter is causal: the estimate at frame k may only use
frames 0..k, because frame k+1 has not happened yet. An offline auto-labeler
has the entire log on disk, so that constraint is self-imposed and expensive.
The Rauch-Tung-Striebel smoother runs the ordinary forward filter, then a
backward recursion that folds all future evidence into every past estimate.

The practical consequence is the whole argument for offboard labeling:

  - A detection at frame 90 sharpens the pose at frame 40. Objects that were
    distant, occluded, or barely resolved early in a track inherit the accuracy
    they earn later, once they are close.
  - Gaps are interpolated by the dynamics rather than by drawing a line. An
    object occluded for 15 frames gets a physically consistent trajectory
    through the gap, with an honest covariance that widens in the middle.
  - Track starts stop being garbage. A forward filter's first few estimates are
    dominated by the prior; the backward pass replaces that prior with
    evidence.

Everything here is dense and Float64. Tracking state vectors are 6 to 10
elements, so the asymptotics of a sparse or factorized implementation never pay
for themselves, and the dense code is the code you can still read at 2am when a
track is diverging.

Numerics worth knowing about:

  - The covariance update is in Joseph form, which stays symmetric positive
    semi-definite under floating point error and under a suboptimal gain. The
    textbook `(I - K H) P` form does not, and its failure mode is a covariance
    that silently loses symmetry over a few thousand frames.
  - `nis` (normalized innovation squared) is returned from every update. It is
    the chi-square gating statistic, and it is what you should feed to an
    assignment solver's gate rather than raw Euclidean distance -- a fast,
    uncertain track deserves a wider gate than a slow, well-observed one.
"""

from std.math import sqrt, abs


# ===----------------------------------------------------------------------===
# Dense matrix
# ===----------------------------------------------------------------------===


struct Matrix(Movable):
    """A dense matrix, row-major.

    One flat `List[Float64]` rather than nested lists: every operation below
    walks rows contiguously.
    """

    var rows: Int
    var cols: Int
    var data: List[Float64]

    def __init__(out self, rows: Int, cols: Int, fill: Float64 = 0.0):
        self.rows = rows
        self.cols = cols
        self.data = List[Float64](length=rows * cols, fill=fill)

    def __init__(
        out self, rows: Int, cols: Int, var data: List[Float64]
    ) raises:
        if len(data) != rows * cols:
            raise Error(
                "Matrix: expected ",
                rows * cols,
                " entries for ",
                rows,
                "x",
                cols,
                ", got ",
                len(data),
            )
        self.rows = rows
        self.cols = cols
        self.data = data^

    @staticmethod
    def identity(n: Int) -> Matrix:
        var m = Matrix(n, n)
        for i in range(n):
            m.data[i * n + i] = 1.0
        return m^

    @staticmethod
    def diagonal(var values: List[Float64]) -> Matrix:
        var n = len(values)
        var m = Matrix(n, n)
        for i in range(n):
            m.data[i * n + i] = values[i]
        return m^

    def get(self, r: Int, c: Int) -> Float64:
        return self.data[r * self.cols + c]

    def set(mut self, r: Int, c: Int, value: Float64):
        self.data[r * self.cols + c] = value

    def copy(self) -> Matrix:
        var out = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            out.data[i] = self.data[i]
        return out^

    def transposed(self) -> Matrix:
        var out = Matrix(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                out.data[c * self.rows + r] = self.data[r * self.cols + c]
        return out^

    def matmul(self, other: Matrix) raises -> Matrix:
        if self.cols != other.rows:
            raise Error(
                "matmul: shape mismatch, ",
                self.rows,
                "x",
                self.cols,
                " times ",
                other.rows,
                "x",
                other.cols,
            )
        var out = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for k in range(self.cols):
                var a = self.data[i * self.cols + k]
                if a == 0.0:
                    continue
                for j in range(other.cols):
                    out.data[i * other.cols + j] += (
                        a * other.data[k * other.cols + j]
                    )
        return out^

    def matvec(self, x: List[Float64]) raises -> List[Float64]:
        if self.cols != len(x):
            raise Error(
                "matvec: ", self.rows, "x", self.cols, " times ", len(x)
            )
        var out = List[Float64](length=self.rows, fill=0.0)
        for i in range(self.rows):
            var acc = 0.0
            for j in range(self.cols):
                acc += self.data[i * self.cols + j] * x[j]
            out[i] = acc
        return out^

    def add(self, other: Matrix) raises -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            raise Error("add: shape mismatch")
        var out = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            out.data[i] = self.data[i] + other.data[i]
        return out^

    def sub(self, other: Matrix) raises -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            raise Error("sub: shape mismatch")
        var out = Matrix(self.rows, self.cols)
        for i in range(len(self.data)):
            out.data[i] = self.data[i] - other.data[i]
        return out^

    def symmetrized(self) -> Matrix:
        """Average with the transpose. Cheap insurance against drift."""
        var out = Matrix(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                out.data[r * self.cols + c] = 0.5 * (
                    self.data[r * self.cols + c] + self.data[c * self.cols + r]
                )
        return out^

    def inverse(self) raises -> Matrix:
        """Gauss-Jordan with partial pivoting.

        Only ever called on the innovation covariance, which is measurement
        sized -- 2 or 3 for a position update. A Cholesky solve would be
        marginally faster and no more accurate at this size.
        """
        if self.rows != self.cols:
            raise Error("inverse: matrix is not square")
        var n = self.rows
        var a = self.copy()
        var inv = Matrix.identity(n)

        for col in range(n):
            var pivot_row = col
            var best = abs(a.get(col, col))
            for r in range(col + 1, n):
                var mag = abs(a.get(r, col))
                if mag > best:
                    best = mag
                    pivot_row = r
            if best < 1e-300:
                raise Error("inverse: matrix is singular at column ", col)

            if pivot_row != col:
                for c in range(n):
                    var t = a.get(col, c)
                    a.set(col, c, a.get(pivot_row, c))
                    a.set(pivot_row, c, t)
                    var ti = inv.get(col, c)
                    inv.set(col, c, inv.get(pivot_row, c))
                    inv.set(pivot_row, c, ti)

            var pivot = a.get(col, col)
            for c in range(n):
                a.set(col, c, a.get(col, c) / pivot)
                inv.set(col, c, inv.get(col, c) / pivot)

            for r in range(n):
                if r == col:
                    continue
                var factor = a.get(r, col)
                if factor == 0.0:
                    continue
                for c in range(n):
                    a.set(r, c, a.get(r, c) - factor * a.get(col, c))
                    inv.set(r, c, inv.get(r, c) - factor * inv.get(col, c))

        return inv^


# ===----------------------------------------------------------------------===
# State
# ===----------------------------------------------------------------------===


struct GaussianState(Movable):
    """A Gaussian belief over the state: mean vector and covariance."""

    var mean: List[Float64]
    var covariance: Matrix

    def __init__(
        out self, var mean: List[Float64], var covariance: Matrix
    ) raises:
        if covariance.rows != len(mean) or covariance.cols != len(mean):
            raise Error(
                "GaussianState: mean of length ",
                len(mean),
                " with ",
                covariance.rows,
                "x",
                covariance.cols,
                " covariance",
            )
        self.mean = mean^
        self.covariance = covariance^

    def dim(self) -> Int:
        return len(self.mean)

    def copy(self) raises -> GaussianState:
        var m = List[Float64](length=len(self.mean), fill=0.0)
        for i in range(len(self.mean)):
            m[i] = self.mean[i]
        return GaussianState(m^, self.covariance.copy())

    def variance(self, i: Int) -> Float64:
        return self.covariance.get(i, i)

    def stddev(self, i: Int) -> Float64:
        return sqrt(self.covariance.get(i, i))


struct UpdateResult(Movable):
    """A posterior plus the diagnostics you need to trust or reject it."""

    var state: GaussianState
    var innovation: List[Float64]
    """Measurement minus predicted measurement."""
    var nis: Float64
    """Normalized innovation squared: `y' S^-1 y`.

    Chi-square distributed with `len(z)` degrees of freedom when the filter is
    consistent. Use it as the gate for data association, and watch its running
    average: a mean far from `len(z)` means the noise model is wrong, not that
    the tracker is unlucky.
    """

    def __init__(
        out self,
        var state: GaussianState,
        var innovation: List[Float64],
        nis: Float64,
    ):
        self.state = state^
        self.innovation = innovation^
        self.nis = nis


# ===----------------------------------------------------------------------===
# Motion and measurement models
# ===----------------------------------------------------------------------===


def constant_velocity(dims: Int, dt: Float64) -> Matrix:
    """State transition for a constant-velocity model.

    State layout is position block then velocity block:
    `[p_0..p_{d-1}, v_0..v_{d-1}]`, so `dims=3` gives the 6-state 3-D tracker.
    """
    var n = 2 * dims
    var f = Matrix.identity(n)
    for i in range(dims):
        f.data[i * n + (dims + i)] = dt
    return f^


def white_noise_acceleration(
    dims: Int, dt: Float64, accel_var: Float64
) -> Matrix:
    """Process noise for a constant-velocity model driven by white acceleration.

    The classic piecewise-white-noise `Q`: an unmodeled acceleration of
    variance `accel_var` acting over `dt` produces position variance
    `dt^4/4`, velocity variance `dt^2`, and cross term `dt^3/2`.

    Pick `accel_var` from physics, not by tuning against a metric. For a
    tracked excavator swinging its house, peak angular acceleration at the
    bucket is a number you can measure once and reuse.
    """
    var n = 2 * dims
    var q = Matrix(n, n)
    var dt2 = dt * dt
    var dt3 = dt2 * dt
    var dt4 = dt2 * dt2
    for i in range(dims):
        var p = i
        var v = dims + i
        q.data[p * n + p] = 0.25 * dt4 * accel_var
        q.data[p * n + v] = 0.5 * dt3 * accel_var
        q.data[v * n + p] = 0.5 * dt3 * accel_var
        q.data[v * n + v] = dt2 * accel_var
    return q^


def position_measurement(dims: Int) -> Matrix:
    """Observation matrix for a detector that reports position only."""
    var h = Matrix(dims, 2 * dims)
    for i in range(dims):
        h.data[i * (2 * dims) + i] = 1.0
    return h^


# ===----------------------------------------------------------------------===
# Filter
# ===----------------------------------------------------------------------===


def predict(state: GaussianState, f: Matrix, q: Matrix) raises -> GaussianState:
    """Propagate the belief forward: `x = F x`, `P = F P F' + Q`."""
    var mean = f.matvec(state.mean)
    var cov = f.matmul(state.covariance).matmul(f.transposed()).add(q)
    return GaussianState(mean^, cov.symmetrized())


def update(
    state: GaussianState, z: List[Float64], h: Matrix, r: Matrix
) raises -> UpdateResult:
    """Fold in one measurement.

    Joseph form: `P = (I - K H) P (I - K H)' + K R K'`. More arithmetic than
    `(I - K H) P`, and worth it -- the Joseph form stays symmetric PSD even
    when the gain is not exactly optimal, which is the regime you are in the
    moment anything about `R` is approximate.
    """
    if len(z) != h.rows:
        raise Error(
            "update: measurement of length ",
            len(z),
            " against ",
            h.rows,
            "-row observation matrix",
        )

    var n = state.dim()
    var ht = h.transposed()

    # Innovation and its covariance.
    var predicted_z = h.matvec(state.mean)
    var y = List[Float64](length=len(z), fill=0.0)
    for i in range(len(z)):
        y[i] = z[i] - predicted_z[i]
    var s = h.matmul(state.covariance).matmul(ht).add(r).symmetrized()
    var s_inv = s.inverse()

    # Kalman gain.
    var k = state.covariance.matmul(ht).matmul(s_inv)

    # Posterior mean.
    var correction = k.matvec(y)
    var mean = List[Float64](length=n, fill=0.0)
    for i in range(n):
        mean[i] = state.mean[i] + correction[i]

    # Posterior covariance, Joseph form.
    var ikh = Matrix.identity(n).sub(k.matmul(h))
    var cov = (
        ikh.matmul(state.covariance)
        .matmul(ikh.transposed())
        .add(k.matmul(r).matmul(k.transposed()))
    )

    # Normalized innovation squared.
    var sy = s_inv.matvec(y)
    var nis = 0.0
    for i in range(len(y)):
        nis += y[i] * sy[i]

    return UpdateResult(GaussianState(mean^, cov.symmetrized()), y^, nis)


# ===----------------------------------------------------------------------===
# Track-level: forward filter then RTS backward pass
# ===----------------------------------------------------------------------===


struct Observation(Movable):
    """One frame of a track. A frame with no detection is still a frame."""

    var dt: Float64
    """Seconds since the previous frame. Ignored on the first."""
    var has_measurement: Bool
    var z: List[Float64]

    def __init__(out self, dt: Float64, var z: List[Float64]):
        self.dt = dt
        self.has_measurement = True
        self.z = z^

    @staticmethod
    def gap(dt: Float64) -> Observation:
        """A frame where the object was occluded or the detector missed it."""
        var o = Observation(dt, List[Float64]())
        o.has_measurement = False
        return o^


struct SmoothedTrack(Movable):
    """The output of one offline refinement pass over a track."""

    var states: List[GaussianState]
    """One smoothed belief per input frame, including the gaps."""
    var nis: List[Float64]
    """Per-frame gating statistic from the forward pass. `-1.0` on gap frames."""
    var num_measurements: Int

    def __init__(
        out self,
        var states: List[GaussianState],
        var nis: List[Float64],
        num_measurements: Int,
    ):
        self.states = states^
        self.nis = nis^
        self.num_measurements = num_measurements


def filter_forward(
    prior: GaussianState,
    observations: List[Observation],
    dims: Int,
    accel_var: Float64,
    r: Matrix,
    mut priors: List[GaussianState],
    mut posteriors: List[GaussianState],
    mut transitions: List[Matrix],
    mut nis: List[Float64],
) raises -> Int:
    """Causal pass. Records the per-step priors the backward pass needs.

    The smoother cannot run from posteriors alone -- it needs, at each step,
    the predicted belief *before* that step's measurement, plus the transition
    that produced it. Storing them here is what keeps `rts_smooth` a pure
    backward recursion.
    """
    var h = position_measurement(dims)
    var current = prior.copy()
    var count = 0

    for i in range(len(observations)):
        var obs_dt = observations[i].dt
        var f = constant_velocity(dims, obs_dt) if i > 0 else Matrix.identity(
            2 * dims
        )
        var q = white_noise_acceleration(
            dims, obs_dt, accel_var
        ) if i > 0 else Matrix(2 * dims, 2 * dims)

        var predicted = predict(current, f, q)
        priors.append(predicted.copy())
        transitions.append(f^)

        if observations[i].has_measurement:
            var res = update(predicted, observations[i].z, h, r)
            nis.append(res.nis)
            count += 1
            posteriors.append(res.state.copy())
        else:
            # No evidence: the posterior is the prior, and the covariance is
            # allowed to grow. This is how an occlusion gap gets an honest
            # uncertainty rather than a confident straight line.
            nis.append(-1.0)
            posteriors.append(predicted^)
        current = posteriors[len(posteriors) - 1].copy()

    return count


def rts_smooth(
    priors: List[GaussianState],
    posteriors: List[GaussianState],
    transitions: List[Matrix],
) raises -> List[GaussianState]:
    """Rauch-Tung-Striebel backward recursion.

    For each step, working backwards:

        C  = P_post F' P_prior(k+1)^-1
        x  = x_post + C (x_smooth(k+1) - x_prior(k+1))
        P  = P_post + C (P_smooth(k+1) - P_prior(k+1)) C'

    `C` is the smoother gain: it says how much of the future's correction to
    credit to this step, and it is large exactly where this step was uncertain
    and the transition was informative.
    """
    var n = len(posteriors)
    var smoothed = List[GaussianState]()
    if n == 0:
        return smoothed^

    for i in range(n):
        smoothed.append(posteriors[i].copy())

    # The last posterior is already conditioned on everything, so the
    # recursion seeds there and walks back.
    for step in range(n - 2, -1, -1):
        ref f_next = transitions[step + 1]
        ref prior_next = priors[step + 1]
        ref post = posteriors[step]

        var c = post.covariance.matmul(f_next.transposed()).matmul(
            prior_next.covariance.inverse()
        )

        var mean_diff = List[Float64](length=post.dim(), fill=0.0)
        for i in range(post.dim()):
            mean_diff[i] = smoothed[step + 1].mean[i] - prior_next.mean[i]
        var mean_corr = c.matvec(mean_diff)
        var mean = List[Float64](length=post.dim(), fill=0.0)
        for i in range(post.dim()):
            mean[i] = post.mean[i] + mean_corr[i]

        var cov_diff = smoothed[step + 1].covariance.sub(prior_next.covariance)
        var cov = post.covariance.add(c.matmul(cov_diff).matmul(c.transposed()))

        smoothed[step] = GaussianState(mean^, cov.symmetrized())

    return smoothed^


def smooth_track(
    prior: GaussianState,
    observations: List[Observation],
    dims: Int,
    accel_var: Float64,
    measurement_var: Float64,
) raises -> SmoothedTrack:
    """Refine one track offline: forward filter, then RTS backward pass.

    This is the entry point an auto-labeling pipeline calls once per tracklet
    after association has run. Gap frames are filled by the dynamics, and every
    frame -- including the first -- is conditioned on the whole sequence.
    """
    var r = Matrix.identity(dims)
    for i in range(dims):
        r.set(i, i, measurement_var)

    var priors = List[GaussianState]()
    var posteriors = List[GaussianState]()
    var transitions = List[Matrix]()
    var nis = List[Float64]()

    var count = filter_forward(
        prior,
        observations,
        dims,
        accel_var,
        r,
        priors,
        posteriors,
        transitions,
        nis,
    )
    var smoothed = rts_smooth(priors, posteriors, transitions)
    return SmoothedTrack(smoothed^, nis^, count)
