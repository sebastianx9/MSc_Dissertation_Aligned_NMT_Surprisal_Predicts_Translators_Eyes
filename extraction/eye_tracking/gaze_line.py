"""Trial-level text-line estimation for the EMMT gaze recordings.

The EMMT stimulus contains one horizontal sentence, but gaze coordinates can
shift and tilt after calibration drift.  This module therefore estimates a
robust line separately for each trial-stage instead of selecting fixations by
a fixed screen-y window.  The approach follows the regression-based family of
post-hoc drift correction methods used in reading research.

All fixation bouts remain in temporal order.  Bouts outside the estimated
sentence region, and bouts without usable coordinates, are later represented
by OFFTEXT/UNKNOWN sentinels so they can terminate a first-pass sequence.
"""

from dataclasses import asdict, dataclass
import csv
import math
from statistics import median

from timestamp_utils import ts_to_seconds


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 1024
MIN_FIX_MS = 20
X_ESTIMATION_MARGIN_PX = 30
X_MAPPING_MARGIN_PX = 30
MAX_ABS_SLOPE = 0.25
INITIAL_HALF_WIDTH_PX = 30
MIN_FINAL_HALF_WIDTH_PX = 24
MAX_FINAL_HALF_WIDTH_PX = 45
N_X_BINS = 10
MIN_MODE_SCORE_RATIO = 2.0


@dataclass(frozen=True)
class FixationBout:
    """One consecutive fixation event, including coordinate-missing bouts."""

    sequence: int
    start_s: float
    end_s: float
    mean_x: float | None
    mean_y: float | None
    duration_ms: float
    coordinate_fraction: float = 1.0
    x_mad: float | None = 0.0
    y_mad: float | None = 0.0


@dataclass(frozen=True)
class LineModel:
    """A trial-level sentence line in screen coordinates."""

    alpha: float
    beta: float
    x_center: float
    half_width: float
    fit_source: str

    def predict(self, x):
        return self.alpha + self.beta * (x - self.x_center)


@dataclass
class LineDiagnostic:
    """Audit information for one trial-stage line estimate."""

    status: str
    reason: str
    fit_source: str
    line_y: float | None
    alpha: float | None
    beta: float | None
    half_width_px: float | None
    residual_mad_px: float | None
    n_raw_bouts: int
    n_candidate_bouts: int
    n_line_bouts: int
    n_x_bins: int
    n_word_aois: int
    candidate_duration_ms: float
    line_duration_ms: float
    line_bout_share: float | None
    line_duration_share: float | None
    line_x_span_px: float | None
    robust_x_span_norm: float | None
    mode_score_ratio: float | None
    independent_failure_reason: str
    read_alpha: float | None
    read_beta: float | None
    translation_read_shift_px: float | None
    n_mapped_word_bouts: int
    n_offtext_bouts: int
    n_unknown_bouts: int
    mapped_word_duration_ms: float
    offtext_duration_ms: float
    unknown_duration_ms: float
    quality_flags: str
    review_flags: str

    def to_dict(self):
        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, float) and math.isfinite(value):
                values[key] = round(value, 4)
        return values


@dataclass(frozen=True)
class _CandidateLine:
    alpha: float
    beta: float
    inlier_sequences: frozenset
    score: float

    def predict(self, x, x_center):
        return self.alpha + self.beta * (x - x_center)


def _mad(values):
    if not values:
        return 0.0
    centre = median(values)
    return median([abs(value - centre) for value in values])


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def read_fixation_bouts(filepath, min_fix_ms=MIN_FIX_MS):
    """Read a gaze CSV and return every >= ``min_fix_ms`` fixation bout."""

    samples = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = ts_to_seconds(row["TimeStamp"])
                event = row["Event"].strip().lower()
            except (AttributeError, KeyError, ValueError):
                continue
            try:
                raw_x = row["X"]
                x = float(raw_x.strip()) if raw_x and raw_x.strip() else None
            except (AttributeError, KeyError, ValueError):
                x = None
            try:
                raw_y = row["Y"]
                y = float(raw_y.strip()) if raw_y and raw_y.strip() else None
            except (AttributeError, KeyError, ValueError):
                y = None
            samples.append((timestamp, x, y, event))

    if len(samples) < 2:
        return []

    positive_diffs = sorted(
        samples[i + 1][0] - samples[i][0]
        for i in range(len(samples) - 1)
        if samples[i + 1][0] > samples[i][0]
    )
    sample_interval = (
        positive_diffs[len(positive_diffs) // 2] if positive_diffs else 0.0005
    )

    bouts = []
    bout_samples = []

    def close_bout():
        nonlocal bout_samples
        if not bout_samples:
            return
        duration_ms = (
            (bout_samples[-1][0] - bout_samples[0][0]) + sample_interval
        ) * 1000
        if duration_ms < min_fix_ms:
            return
        valid_xy = [
            (x, y) for _, x, y, _ in bout_samples if x is not None and y is not None
        ]
        valid_x = [x for x, _ in valid_xy]
        valid_y = [y for _, y in valid_xy]
        bouts.append(
            FixationBout(
                sequence=len(bouts),
                start_s=bout_samples[0][0],
                end_s=bout_samples[-1][0] + sample_interval,
                mean_x=median(valid_x) if valid_x else None,
                mean_y=median(valid_y) if valid_y else None,
                duration_ms=duration_ms,
                coordinate_fraction=len(valid_xy) / len(bout_samples),
                x_mad=_mad(valid_x) if valid_x else None,
                y_mad=_mad(valid_y) if valid_y else None,
            )
        )

    in_fixation = False
    for sample in samples:
        if sample[3] == "fixation":
            if not in_fixation:
                bout_samples = []
                in_fixation = True
            bout_samples.append(sample)
        elif in_fixation:
            close_bout()
            bout_samples = []
            in_fixation = False
    if in_fixation:
        close_bout()
    return bouts


def x_to_word_index(x, ranges, mapping_margin=X_MAPPING_MARGIN_PX):
    """Map x to a word ROI; return -1 beyond the bounded sentence region."""

    if x is None or not ranges:
        return -1
    left = ranges[0][0]
    right = ranges[-1][1]
    if x < left - mapping_margin or x > right + mapping_margin:
        return -1
    for index, (x0, x1, _) in enumerate(ranges):
        if x0 <= x <= x1:
            return index
    # Divide an inter-word space at its own midpoint. Nearest word centres
    # shift this boundary towards a short word when adjacent word widths
    # differ, even though the rendered space itself is symmetric.
    for index in range(len(ranges) - 1):
        left_edge = ranges[index][1]
        right_edge = ranges[index + 1][0]
        if left_edge < x < right_edge:
            return index if x <= (left_edge + right_edge) / 2 else index + 1
    return 0 if x < left else len(ranges) - 1


def _strict_word_index(x, ranges):
    """Word mapping used only for line-fit QC, without an edge margin."""

    if x is None or not ranges or x < ranges[0][0] or x > ranges[-1][1]:
        return -1
    return x_to_word_index(x, ranges, mapping_margin=0)


def _candidate_bouts(bouts, left, right):
    return [
        bout
        for bout in bouts
        if bout.mean_x is not None
        and bout.mean_y is not None
        and left - X_ESTIMATION_MARGIN_PX
        <= bout.mean_x
        <= right + X_ESTIMATION_MARGIN_PX
        and 0 <= bout.mean_x <= SCREEN_WIDTH
        and 0 <= bout.mean_y <= SCREEN_HEIGHT
    ]


def _x_fit_stats(inliers, left, right, ranges):
    width = max(right - left, 1.0)
    strict = [bout for bout in inliers if left <= bout.mean_x <= right]
    xs = [bout.mean_x for bout in strict]
    bins = {
        min(N_X_BINS - 1, max(0, int((x - left) / width * N_X_BINS)))
        for x in xs
    }
    q10 = _quantile(xs, 0.10)
    q90 = _quantile(xs, 0.90)
    robust_span = (q90 - q10) / width if q10 is not None else 0.0
    word_aois = {
        _strict_word_index(bout.mean_x, ranges)
        for bout in strict
        if _strict_word_index(bout.mean_x, ranges) >= 0
    }
    return len(bins), robust_span, len(word_aois)


def _candidate_score(inliers, left, right, ranges):
    n_bins, robust_span, _ = _x_fit_stats(inliers, left, right, ranges)
    bin_coverage = n_bins / N_X_BINS
    return (
        len(inliers)
        * min(bin_coverage / 0.50, 1.0)
        * min(robust_span / 0.50, 1.0)
    )


def _windows_for_slope(candidates, beta, x_center, left, right, ranges):
    adjusted = sorted(
        (
            (
                bout.mean_y - beta * (bout.mean_x - x_center),
                bout,
            )
            for bout in candidates
        ),
        key=lambda item: (item[0], item[1].sequence),
    )
    raw_windows = []
    start = 0
    for end in range(len(adjusted)):
        while (
            adjusted[end][0] - adjusted[start][0]
            > 2 * INITIAL_HALF_WIDTH_PX
        ):
            start += 1
        window = adjusted[start : end + 1]
        if len(window) < 2:
            continue
        alpha = median([value for value, _ in window])
        inliers = [
            bout
            for value, bout in adjusted
            if abs(value - alpha) <= INITIAL_HALF_WIDTH_PX
        ]
        raw_windows.append(
            (len(inliers), alpha, inliers, _candidate_score(inliers, left, right, ranges))
        )

    # Windows overlap heavily. Retain at most four distinct intercept modes for
    # each slope before comparing modes across slopes.
    selected = []
    for _, alpha, inliers, score in sorted(
        raw_windows, key=lambda item: (item[0], item[3]), reverse=True
    ):
        if any(abs(alpha - existing.alpha) < INITIAL_HALF_WIDTH_PX for existing in selected):
            continue
        selected.append(
            _CandidateLine(
                alpha=alpha,
                beta=beta,
                inlier_sequences=frozenset(bout.sequence for bout in inliers),
                score=score,
            )
        )
        if len(selected) == 4:
            break
    return selected


def _line_candidates(candidates, left, right, ranges):
    x_center = (left + right) / 2
    slopes = [(-MAX_ABS_SLOPE + step * 0.025) for step in range(21)]
    models = []
    for beta in slopes:
        models.extend(
            _windows_for_slope(
                candidates, beta, x_center, left, right, ranges
            )
        )
    return models


def _linear_refit(candidates, seed, left, right, fit_source):
    x_center = (left + right) / 2
    selected = [
        bout for bout in candidates if bout.sequence in seed.inlier_sequences
    ]
    beta = seed.beta
    alpha = seed.alpha

    for _ in range(3):
        if len(selected) >= 2:
            xs = [bout.mean_x - x_center for bout in selected]
            ys = [bout.mean_y for bout in selected]
            x_bar = sum(xs) / len(xs)
            y_bar = sum(ys) / len(ys)
            denominator = sum((x - x_bar) ** 2 for x in xs)
            if denominator > 0:
                beta = sum(
                    (x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)
                ) / denominator
                beta = max(-MAX_ABS_SLOPE, min(MAX_ABS_SLOPE, beta))
            alpha = median(
                [
                    bout.mean_y - beta * (bout.mean_x - x_center)
                    for bout in selected
                ]
            )
        residuals = [
            bout.mean_y - (alpha + beta * (bout.mean_x - x_center))
            for bout in selected
        ]
        residual_mad = _mad(residuals)
        half_width = max(
            MIN_FINAL_HALF_WIDTH_PX,
            min(MAX_FINAL_HALF_WIDTH_PX, 2.5 * 1.4826 * residual_mad),
        )
        selected = [
            bout
            for bout in candidates
            if abs(
                bout.mean_y - (alpha + beta * (bout.mean_x - x_center))
            )
            <= half_width
        ]

    residuals = [
        bout.mean_y - (alpha + beta * (bout.mean_x - x_center))
        for bout in selected
    ]
    residual_mad = _mad(residuals)
    half_width = max(
        MIN_FINAL_HALF_WIDTH_PX,
        min(MAX_FINAL_HALF_WIDTH_PX, 2.5 * 1.4826 * residual_mad),
    )
    model = LineModel(alpha, beta, x_center, half_width, fit_source)
    inliers = [
        bout
        for bout in candidates
        if abs(bout.mean_y - model.predict(bout.mean_x)) <= half_width
    ]
    return model, inliers, residual_mad


def _fixed_slope_refit(candidates, seed, prior_model):
    """Refit a prior-supported line without ever freeing the READ slope.

    The intercept, residual MAD, band width, and membership are iterated as a
    single fixed-slope model. This avoids fitting a free TRANSLATE slope and
    then replacing it with the READ slope after the band has already been
    chosen.
    """

    selected_sequences = set(seed.inlier_sequences)
    alpha = seed.alpha
    residual_mad = 0.0
    half_width = MIN_FINAL_HALF_WIDTH_PX
    seen_memberships = {}
    iteration_states = []

    for _ in range(50):
        membership = frozenset(selected_sequences)
        seen_memberships.setdefault(membership, len(iteration_states))
        selected = [
            bout for bout in candidates if bout.sequence in selected_sequences
        ]
        if not selected:
            break
        alpha = median(
            bout.mean_y
            - prior_model.beta * (bout.mean_x - prior_model.x_center)
            for bout in selected
        )
        residuals = [
            bout.mean_y
            - (alpha + prior_model.beta * (bout.mean_x - prior_model.x_center))
            for bout in selected
        ]
        residual_mad = _mad(residuals)
        half_width = max(
            MIN_FINAL_HALF_WIDTH_PX,
            min(MAX_FINAL_HALF_WIDTH_PX, 2.5 * 1.4826 * residual_mad),
        )
        updated_sequences = {
            bout.sequence
            for bout in candidates
            if abs(
                bout.mean_y
                - (
                    alpha
                    + prior_model.beta * (bout.mean_x - prior_model.x_center)
                )
            )
            <= half_width
        }
        iteration_states.append((membership, half_width))
        if updated_sequences == selected_sequences:
            break
        updated_membership = frozenset(updated_sequences)
        if updated_membership in seen_memberships:
            # Adaptive MAD bands can occasionally alternate between two
            # memberships when one bout sits exactly at the changing edge.
            # Resolve that discrete cycle deterministically and
            # conservatively: retain the intersection of the cycle
            # memberships and the narrowest width encountered. This avoids an
            # arbitrary dependence on iteration parity and does not admit the
            # boundary bout that caused the oscillation.
            cycle_start = seen_memberships[updated_membership]
            cycle_states = iteration_states[cycle_start:]
            cycle_memberships = [
                set(state) for state, _ in cycle_states
            ] + [set(updated_membership)]
            cycle_membership = set.intersection(*cycle_memberships)
            half_width = min(width for _, width in cycle_states)

            # With width fixed, update the intercept and membership to a
            # geometric fixed point. The intersection/minimum-width start is
            # deliberately conservative if a second discrete cycle were ever
            # encountered.
            fixed_seen = []
            for _ in range(50):
                cycle_selected = [
                    bout
                    for bout in candidates
                    if bout.sequence in cycle_membership
                ]
                if not cycle_selected:
                    break
                alpha = median(
                    bout.mean_y
                    - prior_model.beta
                    * (bout.mean_x - prior_model.x_center)
                    for bout in cycle_selected
                )
                classified = {
                    bout.sequence
                    for bout in candidates
                    if abs(
                        bout.mean_y
                        - (
                            alpha
                            + prior_model.beta
                            * (bout.mean_x - prior_model.x_center)
                        )
                    )
                    <= half_width
                }
                if classified == cycle_membership:
                    break
                if classified in fixed_seen:
                    cycle_membership &= classified
                    break
                fixed_seen.append(set(cycle_membership))
                cycle_membership = classified

            cycle_selected = [
                bout
                for bout in candidates
                if bout.sequence in cycle_membership
            ]
            if cycle_selected:
                alpha = median(
                    bout.mean_y
                    - prior_model.beta
                    * (bout.mean_x - prior_model.x_center)
                    for bout in cycle_selected
                )
            model = LineModel(
                alpha=alpha,
                beta=prior_model.beta,
                x_center=prior_model.x_center,
                half_width=half_width,
                fit_source="read_prior",
            )
            inliers = [
                bout
                for bout in candidates
                if abs(bout.mean_y - model.predict(bout.mean_x))
                <= model.half_width
            ]
            residual_mad = _mad(
                [bout.mean_y - model.predict(bout.mean_x) for bout in inliers]
            )
            return model, inliers, residual_mad, True
        selected_sequences = updated_sequences

    inliers = [
        bout for bout in candidates if bout.sequence in selected_sequences
    ]
    if inliers:
        # Recompute all parameters once from the final membership. The loop
        # normally reaches a fixed point; this final pass keeps the returned
        # width and residual statistic internally consistent in every case.
        alpha = median(
            bout.mean_y
            - prior_model.beta * (bout.mean_x - prior_model.x_center)
            for bout in inliers
        )
        residuals = [
            bout.mean_y
            - (alpha + prior_model.beta * (bout.mean_x - prior_model.x_center))
            for bout in inliers
        ]
        residual_mad = _mad(residuals)
        half_width = max(
            MIN_FINAL_HALF_WIDTH_PX,
            min(MAX_FINAL_HALF_WIDTH_PX, 2.5 * 1.4826 * residual_mad),
        )

    model = LineModel(
        alpha=alpha,
        beta=prior_model.beta,
        x_center=prior_model.x_center,
        half_width=half_width,
        fit_source="read_prior",
    )
    inliers = [
        bout
        for bout in candidates
        if abs(bout.mean_y - model.predict(bout.mean_x)) <= model.half_width
    ]
    residual_mad = _mad(
        [bout.mean_y - model.predict(bout.mean_x) for bout in inliers]
    )
    return model, inliers, residual_mad, False


def _residual_mode_ratio(candidates, model):
    """Return main/competitor density for vertically separated residual bands.

    Candidate RANSAC lines overlap heavily on a single tilted scanpath, so a
    runner-up line is not itself evidence of a second fixation band.  Here the
    competitor is assessed only after the main line has been fitted, using
    equal-bout Gaussian density on residuals at least 100 px from that line.
    """

    residuals = [
        bout.mean_y - model.predict(bout.mean_x) for bout in candidates
    ]
    bandwidth = 25.0

    def density(centre):
        return sum(
            math.exp(-0.5 * ((residual - centre) / bandwidth) ** 2)
            for residual in residuals
        )

    main_centres = [
        residual for residual in residuals if abs(residual) <= model.half_width
    ] or [0.0]
    main_density = max(density(centre) for centre in main_centres)
    alternative_centres = [residual for residual in residuals if abs(residual) >= 100]
    if not alternative_centres:
        return math.inf
    alternative_density = max(density(centre) for centre in alternative_centres)
    return main_density / alternative_density if alternative_density > 0 else math.inf


def _empty_diagnostic(reason, n_raw, n_candidates=0):
    return LineDiagnostic(
        status="rejected",
        reason=reason,
        fit_source="failed",
        line_y=None,
        alpha=None,
        beta=None,
        half_width_px=None,
        residual_mad_px=None,
        n_raw_bouts=n_raw,
        n_candidate_bouts=n_candidates,
        n_line_bouts=0,
        n_x_bins=0,
        n_word_aois=0,
        candidate_duration_ms=0.0,
        line_duration_ms=0.0,
        line_bout_share=None,
        line_duration_share=None,
        line_x_span_px=None,
        robust_x_span_norm=None,
        mode_score_ratio=None,
        independent_failure_reason="",
        read_alpha=None,
        read_beta=None,
        translation_read_shift_px=None,
        n_mapped_word_bouts=0,
        n_offtext_bouts=0,
        n_unknown_bouts=0,
        mapped_word_duration_ms=0.0,
        offtext_duration_ms=0.0,
        unknown_duration_ms=0.0,
        quality_flags=reason,
        review_flags="",
    )


def _diagnostic_for_model(
    model,
    inliers,
    candidates,
    bouts,
    ranges,
    residual_mad,
    mode_ratio,
    stage,
):
    left, right = ranges[0][0], ranges[-1][1]
    n_bins, robust_span, n_word_aois = _x_fit_stats(
        inliers, left, right, ranges
    )
    support = len(inliers) / len(candidates) if candidates else 0.0
    candidate_duration = sum(bout.duration_ms for bout in candidates)
    line_duration = sum(bout.duration_ms for bout in inliers)
    duration_share = (
        line_duration / candidate_duration if candidate_duration > 0 else None
    )
    line_x_span = (
        max(bout.mean_x for bout in inliers)
        - min(bout.mean_x for bout in inliers)
        if inliers
        else 0.0
    )

    required_word_aois = min(4, math.ceil(0.75 * len(ranges)))

    if model.fit_source == "read_prior":
        checks = [
            (len(inliers) >= 3, "fewer_than_three_prior_supported_bouts"),
            (support >= 0.25, "low_prior_supported_share"),
            (n_bins >= 2, "insufficient_prior_x_bins"),
            (robust_span >= 0.15, "insufficient_prior_x_span"),
            (n_word_aois >= 2, "fewer_than_two_prior_word_aois"),
            (residual_mad <= 25, "high_prior_residual_dispersion"),
            (mode_ratio >= MIN_MODE_SCORE_RATIO, "ambiguous_prior_competing_line"),
        ]
    elif stage == "translate":
        checks = [
            (len(inliers) >= 6, "fewer_than_six_line_bouts"),
            (support >= 0.60, "low_line_support"),
            (n_bins >= 5, "insufficient_x_bin_coverage"),
            (robust_span >= 0.45, "insufficient_robust_x_span"),
            (
                n_word_aois >= required_word_aois,
                "insufficient_word_aoi_coverage",
            ),
            (residual_mad <= 25, "high_residual_dispersion"),
            (mode_ratio >= MIN_MODE_SCORE_RATIO, "ambiguous_competing_line"),
        ]
    else:
        checks = [
            (len(inliers) >= 6, "fewer_than_six_line_bouts"),
            (support >= 0.50, "low_line_support"),
            (n_bins >= 4, "insufficient_x_bin_coverage"),
            (robust_span >= 0.40, "insufficient_robust_x_span"),
            (
                n_word_aois >= required_word_aois,
                "insufficient_word_aoi_coverage",
            ),
            (residual_mad <= 25, "high_residual_dispersion"),
            (mode_ratio >= MIN_MODE_SCORE_RATIO, "ambiguous_competing_line"),
        ]

    failures = [reason for passed, reason in checks if not passed]
    return LineDiagnostic(
        status="ok" if not failures else "rejected",
        reason="ok" if not failures else failures[0],
        fit_source=model.fit_source,
        line_y=model.alpha,
        alpha=model.alpha,
        beta=model.beta,
        half_width_px=model.half_width,
        residual_mad_px=residual_mad,
        n_raw_bouts=len(bouts),
        n_candidate_bouts=len(candidates),
        n_line_bouts=len(inliers),
        n_x_bins=n_bins,
        n_word_aois=n_word_aois,
        candidate_duration_ms=candidate_duration,
        line_duration_ms=line_duration,
        line_bout_share=support,
        line_duration_share=duration_share,
        line_x_span_px=line_x_span,
        robust_x_span_norm=robust_span,
        mode_score_ratio=mode_ratio,
        independent_failure_reason="",
        read_alpha=None,
        read_beta=None,
        translation_read_shift_px=None,
        n_mapped_word_bouts=0,
        n_offtext_bouts=0,
        n_unknown_bouts=0,
        mapped_word_duration_ms=0.0,
        offtext_duration_ms=0.0,
        unknown_duration_ms=0.0,
        quality_flags=";".join(failures) if failures else "",
        review_flags="",
    )


def _independent_fit(bouts, ranges, stage):
    left, right = ranges[0][0], ranges[-1][1]
    candidates = _candidate_bouts(bouts, left, right)
    if not bouts:
        return [], None, _empty_diagnostic("no_fixation_bouts", 0)
    if not any(bout.mean_x is not None and bout.mean_y is not None for bout in bouts):
        return [], None, _empty_diagnostic(
            "no_bouts_with_valid_coordinates", len(bouts)
        )
    if len(candidates) < 2:
        return [], None, _empty_diagnostic(
            "fewer_than_two_bouts_near_sentence", len(bouts), len(candidates)
        )

    candidates_lines = _line_candidates(candidates, left, right, ranges)
    if not candidates_lines:
        return [], None, _empty_diagnostic(
            "no_candidate_line", len(bouts), len(candidates)
        )
    best = max(candidates_lines, key=lambda model: model.score)
    model, inliers, residual_mad = _linear_refit(
        candidates, best, left, right, "independent"
    )
    mode_ratio = _residual_mode_ratio(candidates, model)
    diagnostic = _diagnostic_for_model(
        model,
        inliers,
        candidates,
        bouts,
        ranges,
        residual_mad,
        mode_ratio,
        stage,
    )
    return inliers, model, diagnostic


def _read_prior_fit(bouts, ranges, prior_model):
    left, right = ranges[0][0], ranges[-1][1]
    candidates = _candidate_bouts(bouts, left, right)
    if len(candidates) < 3:
        return [], None, _empty_diagnostic(
            "insufficient_bouts_for_read_prior", len(bouts), len(candidates)
        )

    windows = _windows_for_slope(
        candidates,
        prior_model.beta,
        prior_model.x_center,
        left,
        right,
        ranges,
    )
    nearby = [
        window for window in windows if abs(window.alpha - prior_model.alpha) <= 80
    ]
    if not nearby:
        return [], None, _empty_diagnostic(
            "no_mode_near_read_line", len(bouts), len(candidates)
        )
    # Evidence determines the mode; distance to READ acts only as a weak tie
    # breaker for modes with similar scan coverage.
    seed = max(
        nearby,
        key=lambda window: (
            window.score,
            -abs(window.alpha - prior_model.alpha),
        ),
    )
    fixed_slope_seed = _CandidateLine(
        alpha=seed.alpha,
        beta=prior_model.beta,
        inlier_sequences=seed.inlier_sequences,
        score=seed.score,
    )
    model, inliers, residual_mad, cycle_resolved = _fixed_slope_refit(
        candidates, fixed_slope_seed, prior_model
    )
    mode_ratio = _residual_mode_ratio(candidates, model)
    diagnostic = _diagnostic_for_model(
        model,
        inliers,
        candidates,
        bouts,
        ranges,
        residual_mad,
        mode_ratio,
        "translate",
    )
    if abs(model.alpha - prior_model.alpha) > 80:
        failure = "read_prior_shift_exceeds_80px"
        diagnostic.status = "rejected"
        diagnostic.reason = failure
        diagnostic.quality_flags = ";".join(
            filter(None, [diagnostic.quality_flags, failure])
        )
    if cycle_resolved:
        diagnostic.review_flags = "prior_membership_cycle_resolved"
    return inliers, model, diagnostic


def _attach_pair_and_review_metadata(
    diagnostic,
    model,
    prior_model=None,
    independent_failure_reason="",
):
    diagnostic.independent_failure_reason = independent_failure_reason
    flags = [
        flag for flag in diagnostic.review_flags.split(";") if flag
    ]
    if prior_model is not None:
        diagnostic.read_alpha = prior_model.alpha
        diagnostic.read_beta = prior_model.beta
        if model is not None:
            diagnostic.translation_read_shift_px = model.alpha - prior_model.alpha
            if abs(diagnostic.translation_read_shift_px) > 60:
                flags.append("large_read_translate_shift")
    if model is not None:
        if model.fit_source == "read_prior":
            flags.append("read_prior_fallback")
            if diagnostic.n_line_bouts <= 3:
                flags.append("minimal_prior_bout_count")
            if (
                diagnostic.line_bout_share is not None
                and diagnostic.line_bout_share < 0.40
            ):
                flags.append("low_prior_bout_support")
            if (
                diagnostic.line_duration_share is not None
                and diagnostic.line_duration_share < 0.35
            ):
                flags.append("low_prior_duration_support")
            if (
                diagnostic.n_x_bins <= 2
                or diagnostic.n_word_aois <= 2
                or (
                    diagnostic.robust_x_span_norm is not None
                    and diagnostic.robust_x_span_norm < 0.20
                )
            ):
                flags.append("limited_prior_horizontal_coverage")
        if model.half_width >= MAX_FINAL_HALF_WIDTH_PX - 1e-6:
            flags.append("half_width_at_cap")
        if abs(model.beta) >= MAX_ABS_SLOPE - 0.025 - 1e-6:
            flags.append("slope_near_cap")
    if (
        diagnostic.mode_score_ratio is not None
        and math.isfinite(diagnostic.mode_score_ratio)
        and diagnostic.mode_score_ratio < 3.0
    ):
        flags.append("moderate_competing_mode")
    diagnostic.review_flags = ";".join(dict.fromkeys(flags))
    return diagnostic


def fit_text_line(bouts, ranges, stage="read", prior_model=None):
    """Fit and quality-check one READ or TRANSLATE sentence line."""

    inliers, model, diagnostic = _independent_fit(bouts, ranges, stage)
    if diagnostic.status == "ok":
        _attach_pair_and_review_metadata(diagnostic, model, prior_model)
        return inliers, model, diagnostic
    if stage == "translate" and prior_model is not None:
        # A prior cannot recover a trial with no usable gaze evidence. Retain
        # the more informative structural failure instead of replacing it
        # with a generic prior-fallback failure.
        if diagnostic.reason in {
            "no_fixation_bouts",
            "no_bouts_with_valid_coordinates",
            "fewer_than_two_bouts_near_sentence",
        }:
            _attach_pair_and_review_metadata(
                diagnostic,
                model,
                prior_model,
                independent_failure_reason=diagnostic.reason,
            )
            return [], None, diagnostic
        independent_failure_reason = diagnostic.reason
        prior_inliers, prior_line, prior_diagnostic = _read_prior_fit(
            bouts, ranges, prior_model
        )
        _attach_pair_and_review_metadata(
            prior_diagnostic,
            prior_line,
            prior_model,
            independent_failure_reason=independent_failure_reason,
        )
        if prior_diagnostic.status == "ok":
            return prior_inliers, prior_line, prior_diagnostic
        return [], None, prior_diagnostic
    _attach_pair_and_review_metadata(diagnostic, model, prior_model)
    return [], None, diagnostic


def estimate_text_line(bouts, x_min, x_max):
    """Backward-compatible test helper using geometric rather than word QC."""

    pseudo_ranges = [
        (x_min + i * (x_max - x_min) / 6, x_min + (i + 1) * (x_max - x_min) / 6, str(i))
        for i in range(6)
    ]
    inliers, _, diagnostic = fit_text_line(bouts, pseudo_ranges, stage="read")
    return inliers, diagnostic


def extract_trial_bouts(filepath, ranges, stage="read", prior_model=None):
    """Return all bouts, accepted line bouts, diagnostic, and fitted model."""

    bouts = read_fixation_bouts(filepath)
    if not ranges:
        diagnostic = _empty_diagnostic("no_word_ranges", len(bouts))
        return bouts, [], diagnostic, None
    inliers, model, diagnostic = fit_text_line(
        bouts, ranges, stage=stage, prior_model=prior_model
    )
    return bouts, inliers, diagnostic, model


def map_trial_bouts(all_bouts, line_bouts, ranges, diagnostic=None):
    """Map a complete ordered bout sequence to WORD/OFFTEXT/UNKNOWN codes.

    WORD bouts use a non-negative word index, OFFTEXT is -1, and UNKNOWN is
    -2. If a diagnostic is supplied, mapping counts and durations are attached
    to it in place.
    """

    line_sequences = {bout.sequence for bout in line_bouts}
    mapped = []
    for bout in all_bouts:
        if bout.mean_x is None or bout.mean_y is None:
            word_index = -2
        elif bout.sequence in line_sequences:
            word_index = x_to_word_index(bout.mean_x, ranges)
        else:
            word_index = -1
        mapped.append((word_index, bout.duration_ms))

    if diagnostic is not None:
        diagnostic.n_mapped_word_bouts = sum(wi >= 0 for wi, _ in mapped)
        diagnostic.n_offtext_bouts = sum(wi == -1 for wi, _ in mapped)
        diagnostic.n_unknown_bouts = sum(wi == -2 for wi, _ in mapped)
        diagnostic.mapped_word_duration_ms = sum(
            duration for wi, duration in mapped if wi >= 0
        )
        diagnostic.offtext_duration_ms = sum(
            duration for wi, duration in mapped if wi == -1
        )
        diagnostic.unknown_duration_ms = sum(
            duration for wi, duration in mapped if wi == -2
        )
    return mapped


def extract_text_line_bouts(
    filepath,
    ranges,
    stage="read",
    prior_model=None,
    return_model=False,
):
    """Read one trial and retain bouts in its estimated sentence-line band."""

    _, inliers, diagnostic, model = extract_trial_bouts(
        filepath, ranges, stage=stage, prior_model=prior_model
    )
    if return_model:
        return inliers, diagnostic, model
    return inliers, diagnostic
