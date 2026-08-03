"""Timestamp parsing shared by the EMMT gaze extractors."""


def ts_to_seconds(ts_str):
    """Convert an H:M:S timestamp to seconds since midnight.

    EMMT timestamps are not consistently zero-padded. Splitting on colons
    therefore handles zero-padded and unpadded fields, with or without a
    fractional component.
    """
    h, m, sec = ts_str.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)
