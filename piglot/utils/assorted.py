"""Assorted utilities."""


def pretty_time(elapsed_sec: float) -> str:
    """Return a human-readable representation of a given elapsed time

    Parameters
    ----------
    elapsed_sec : float
        Elapsed time, in seconds

    Returns
    -------
    str
        Pretty elapsed time string
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    time_str = ''
    for suffix, factor in mults.items():
        count = elapsed_sec // factor
        if count > 0:
            time_str += str(int(elapsed_sec / factor)) + suffix
        elapsed_sec %= factor
    if time_str == '':
        time_str = f'{elapsed_sec:.5f}s'
    return time_str

def reverse_pretty_time(time_str: str) -> float:
    """Return an elapsed time from its human-readable representation

    Parameters
    ----------
    time_str : str
        Pretty elapsed time string

    Returns
    -------
    str
        Elapsed time, in seconds
    """
    mults = {
        'y': 60*60*24*365,
        'd': 60*60*24,
        'h': 60*60,
        'm': 60,
        's': 1,
    }
    value = 0.0
    for suffix, factor in mults.items():
        left, time_str = time_str.split(suffix)
        value += float(left) * factor
    return value
