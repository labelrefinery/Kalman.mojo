"""Tests for the Kalman filter and RTS smoother.

Run with:

    pixi run test
"""

from std.math import sqrt, abs, log, cos, pi
from std.testing import assert_equal, assert_almost_equal, assert_true

from kalman import (
    Matrix,
    GaussianState,
    Observation,
    SmoothedTrack,
    constant_velocity,
    white_noise_acceleration,
    position_measurement,
    predict,
    update,
    filter_forward,
    rts_smooth,
    smooth_track,
)


struct Lcg(Movable):
    """Deterministic generator, so a failing test reproduces exactly."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_uniform(mut self) -> Float64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return (Float64((self.state >> 11) % 1000000007) + 1.0) / 1000000009.0

    def next_gaussian(mut self) -> Float64:
        var u1 = self.next_uniform()
        var u2 = self.next_uniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def state_of(
    var mean: List[Float64], var_diag: Float64
) raises -> GaussianState:
    var n = len(mean)
    var p = Matrix.identity(n)
    for i in range(n):
        p.set(i, i, var_diag)
    return GaussianState(mean^, p^)


# ===----------------------------------------------------------------------===
# Matrix
# ===----------------------------------------------------------------------===


def test_matmul() raises:
    var a = Matrix(2, 3, List[Float64](length=6, fill=0.0))
    var counter = 1.0
    for r in range(2):
        for c in range(3):
            a.set(r, c, counter)
            counter += 1.0
    var b = a.transposed()
    var ab = a.matmul(b)
    assert_equal(ab.rows, 2)
    assert_equal(ab.cols, 2)
    # [[1,2,3],[4,5,6]] @ its transpose
    assert_almost_equal(ab.get(0, 0), 14.0)
    assert_almost_equal(ab.get(0, 1), 32.0)
    assert_almost_equal(ab.get(1, 1), 77.0)


def test_inverse_round_trip() raises:
    var rng = Lcg(0xBEEF)
    for n in range(1, 7):
        var m = Matrix.identity(n)
        for r in range(n):
            for c in range(n):
                m.set(r, c, m.get(r, c) + 0.3 * rng.next_gaussian())
        # Make it diagonally dominant so it is comfortably invertible.
        for i in range(n):
            m.set(i, i, m.get(i, i) + Float64(n))
        var inv = m.inverse()
        var prod = m.matmul(inv)
        for r in range(n):
            for c in range(n):
                var want = 1.0 if r == c else 0.0
                assert_almost_equal(prod.get(r, c), want, atol=1e-9)


def test_singular_inverse_raises() raises:
    var m = Matrix(2, 2)
    var raised = False
    try:
        _ = m.inverse()
    except:
        raised = True
    assert_true(raised)


def test_constant_velocity_layout() raises:
    var f = constant_velocity(3, 0.5)
    assert_equal(f.rows, 6)
    for i in range(3):
        assert_almost_equal(f.get(i, i), 1.0)
        assert_almost_equal(f.get(i, 3 + i), 0.5)
        assert_almost_equal(f.get(3 + i, 3 + i), 1.0)
        assert_almost_equal(f.get(3 + i, i), 0.0)


# ===----------------------------------------------------------------------===
# Filter
# ===----------------------------------------------------------------------===


def test_predict_moves_by_velocity() raises:
    var s = state_of([0.0, 0.0, 0.0, 2.0, -1.0, 0.5], 1.0)
    var f = constant_velocity(3, 2.0)
    var q = white_noise_acceleration(3, 2.0, 0.1)
    var out = predict(s, f, q)
    assert_almost_equal(out.mean[0], 4.0)
    assert_almost_equal(out.mean[1], -2.0)
    assert_almost_equal(out.mean[2], 1.0)
    # Velocity is unchanged, and uncertainty only ever grows on a predict.
    assert_almost_equal(out.mean[3], 2.0)
    assert_true(out.variance(0) > s.variance(0))


def test_update_pulls_toward_confident_measurement() raises:
    var s = state_of([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100.0)
    var h = position_measurement(3)
    var r = Matrix.identity(3)
    for i in range(3):
        r.set(i, i, 1e-6)
    var res = update(s, [5.0, -3.0, 1.0], h, r)
    assert_almost_equal(res.state.mean[0], 5.0, atol=1e-3)
    assert_almost_equal(res.state.mean[1], -3.0, atol=1e-3)
    assert_true(res.state.variance(0) < s.variance(0))


def test_update_covariance_stays_symmetric() raises:
    var s = state_of([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], 4.0)
    var h = position_measurement(3)
    var r = Matrix.identity(3)
    var res = update(s, [1.5, 2.5, 2.5], h, r)
    for i in range(6):
        for j in range(6):
            assert_almost_equal(
                res.state.covariance.get(i, j),
                res.state.covariance.get(j, i),
                atol=1e-12,
            )


def test_nis_matches_hand_computation() raises:
    # 1-D position-only: P = 3, R = 1, so S = 4. An innovation of 2 gives
    # NIS = 4/4 = 1.
    var p = Matrix.identity(2)
    p.set(0, 0, 3.0)
    p.set(1, 1, 3.0)
    var s = GaussianState([0.0, 0.0], p^)
    var h = position_measurement(1)
    var r = Matrix.identity(1)
    var res = update(s, [2.0], h, r)
    assert_almost_equal(res.innovation[0], 2.0)
    assert_almost_equal(res.nis, 1.0, atol=1e-12)


# ===----------------------------------------------------------------------===
# Smoother
# ===----------------------------------------------------------------------===


def build_track(
    mut rng: Lcg,
    n: Int,
    dt: Float64,
    noise_sd: Float64,
    gap_from: Int,
    gap_to: Int,
) raises -> List[Observation]:
    """A 1-D constant-velocity track: x(t) = 10 + 3t, with a hole in it."""
    var obs = List[Observation]()
    for k in range(n):
        var t = Float64(k) * dt
        if k >= gap_from and k < gap_to:
            obs.append(Observation.gap(dt))
        else:
            var truth = 10.0 + 3.0 * t
            obs.append(
                Observation(dt, [truth + noise_sd * rng.next_gaussian()])
            )
    return obs^


def truth_at(k: Int, dt: Float64) -> Float64:
    return 10.0 + 3.0 * Float64(k) * dt


def test_smoother_last_state_equals_filter_posterior() raises:
    """RTS boundary condition: the final estimate is unchanged by smoothing."""
    var rng = Lcg(1)
    var obs = build_track(rng, 20, 0.1, 0.3, 100, 100)
    var prior = state_of([0.0, 0.0], 1e4)

    var r = Matrix.identity(1)
    r.set(0, 0, 0.09)
    var priors = List[GaussianState]()
    var posteriors = List[GaussianState]()
    var transitions = List[Matrix]()
    var nis = List[Float64]()
    _ = filter_forward(
        prior, obs, 1, 1.0, r, priors, posteriors, transitions, nis
    )
    var smoothed = rts_smooth(priors, posteriors, transitions)

    var last = len(obs) - 1
    assert_almost_equal(
        smoothed[last].mean[0], posteriors[last].mean[0], atol=1e-9
    )
    assert_almost_equal(
        smoothed[last].mean[1], posteriors[last].mean[1], atol=1e-9
    )


def test_smoothing_never_increases_variance() raises:
    """Conditioning on more data cannot make an estimate less certain."""
    var rng = Lcg(7)
    var obs = build_track(rng, 30, 0.1, 0.3, 12, 18)
    var prior = state_of([0.0, 0.0], 1e4)

    var r = Matrix.identity(1)
    r.set(0, 0, 0.09)
    var priors = List[GaussianState]()
    var posteriors = List[GaussianState]()
    var transitions = List[Matrix]()
    var nis = List[Float64]()
    _ = filter_forward(
        prior, obs, 1, 1.0, r, priors, posteriors, transitions, nis
    )
    var smoothed = rts_smooth(priors, posteriors, transitions)

    for k in range(len(obs)):
        for i in range(2):
            assert_true(
                smoothed[k].variance(i) <= posteriors[k].variance(i) + 1e-9
            )


def test_smoothing_beats_filtering_on_rmse() raises:
    """The headline claim of offline refinement, measured.

    Same track, same noise, same model -- the only difference is that the
    smoother is allowed to look ahead. If this ever fails, the backward
    recursion is wrong.
    """
    var rng = Lcg(0xD00D)
    var n = 40
    var dt = 0.1
    var obs = build_track(rng, n, dt, 0.3, 100, 100)
    var prior = state_of([0.0, 0.0], 1e4)

    var r = Matrix.identity(1)
    r.set(0, 0, 0.09)
    var priors = List[GaussianState]()
    var posteriors = List[GaussianState]()
    var transitions = List[Matrix]()
    var nis = List[Float64]()
    _ = filter_forward(
        prior, obs, 1, 1.0, r, priors, posteriors, transitions, nis
    )
    var smoothed = rts_smooth(priors, posteriors, transitions)

    var filtered_sq = 0.0
    var smoothed_sq = 0.0
    for k in range(n):
        var truth = truth_at(k, dt)
        var ef = posteriors[k].mean[0] - truth
        var es = smoothed[k].mean[0] - truth
        filtered_sq += ef * ef
        smoothed_sq += es * es
    var filtered_rmse = sqrt(filtered_sq / Float64(n))
    var smoothed_rmse = sqrt(smoothed_sq / Float64(n))
    assert_true(smoothed_rmse < filtered_rmse)


def test_gap_is_filled_by_dynamics() raises:
    """An occluded stretch gets a physically consistent trajectory."""
    var rng = Lcg(0xFEED)
    var n = 40
    var dt = 0.1
    var gap_from = 15
    var gap_to = 25
    var obs = build_track(rng, n, dt, 0.2, gap_from, gap_to)
    var track = smooth_track(state_of([0.0, 0.0], 1e4), obs, 1, 1.0, 0.04)

    assert_equal(track.num_measurements, n - (gap_to - gap_from))
    assert_equal(len(track.states), n)

    # Inside the hole the estimate still tracks truth closely.
    for k in range(gap_from, gap_to):
        assert_almost_equal(track.states[k].mean[0], truth_at(k, dt), atol=0.35)

    # And the uncertainty is honest: highest in the middle of the gap, lower
    # at frames that were actually observed.
    var mid = (gap_from + gap_to) // 2
    assert_true(
        track.states[mid].variance(0) > track.states[gap_from - 1].variance(0)
    )
    assert_true(
        track.states[mid].variance(0) > track.states[gap_to].variance(0)
    )

    # Gap frames are flagged rather than given a fabricated NIS.
    for k in range(gap_from, gap_to):
        assert_almost_equal(track.nis[k], -1.0)


def test_velocity_is_recovered() raises:
    var rng = Lcg(0xACE)
    var obs = build_track(rng, 60, 0.1, 0.1, 100, 100)
    var track = smooth_track(state_of([0.0, 0.0], 1e4), obs, 1, 0.5, 0.01)
    # True velocity is 3.0 m/s. The smoother sees the whole track, so even the
    # first frame should be close -- that is the part a forward filter cannot do.
    assert_almost_equal(track.states[0].mean[1], 3.0, atol=0.35)
    assert_almost_equal(track.states[30].mean[1], 3.0, atol=0.35)


def test_smoothed_start_beats_filtered_start() raises:
    """Track births are where the smoother earns the most."""
    var rng = Lcg(0x1234)
    var n = 30
    var dt = 0.1
    var obs = build_track(rng, n, dt, 0.3, 100, 100)
    var prior = state_of([0.0, 0.0], 1e4)

    var r = Matrix.identity(1)
    r.set(0, 0, 0.09)
    var priors = List[GaussianState]()
    var posteriors = List[GaussianState]()
    var transitions = List[Matrix]()
    var nis = List[Float64]()
    _ = filter_forward(
        prior, obs, 1, 1.0, r, priors, posteriors, transitions, nis
    )
    var smoothed = rts_smooth(priors, posteriors, transitions)

    # Velocity at frame 0 is unobservable to a causal filter -- it has seen
    # exactly one position. The smoother has seen thirty.
    var filtered_err = abs(posteriors[0].mean[1] - 3.0)
    var smoothed_err = abs(smoothed[0].mean[1] - 3.0)
    assert_true(smoothed_err < filtered_err)


def test_empty_track() raises:
    var track = smooth_track(
        state_of([0.0, 0.0], 1.0), List[Observation](), 1, 1.0, 1.0
    )
    assert_equal(len(track.states), 0)
    assert_equal(track.num_measurements, 0)


def main() raises:
    test_matmul()
    test_inverse_round_trip()
    test_singular_inverse_raises()
    test_constant_velocity_layout()
    test_predict_moves_by_velocity()
    test_update_pulls_toward_confident_measurement()
    test_update_covariance_stays_symmetric()
    test_nis_matches_hand_computation()
    test_smoother_last_state_equals_filter_posterior()
    test_smoothing_never_increases_variance()
    test_smoothing_beats_filtering_on_rmse()
    test_gap_is_filled_by_dynamics()
    test_velocity_is_recovered()
    test_smoothed_start_beats_filtered_start()
    test_empty_track()
    print("kalman: all tests passed")
