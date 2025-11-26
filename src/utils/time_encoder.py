from datetime import datetime
import calendar
import numpy as np


SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
SECONDS_IN_DAY = HOURS_IN_DAY * MINUTES_IN_HOUR * SECONDS_IN_MINUTE


def encode_time(time: datetime):
    days_in_year = 366 if calendar.isleap(time.year) else 365
    doy = int(time.strftime("%j"))
    doy_sin = np.sin(2 * np.pi * doy / days_in_year)
    doy_cos = np.cos(2 * np.pi * doy / days_in_year)

    tod = (
        time.hour * MINUTES_IN_HOUR * SECONDS_IN_MINUTE
        + time.minute * SECONDS_IN_MINUTE
        + time.second
    )
    tod_sin = np.sin(2 * np.pi * tod / SECONDS_IN_DAY)
    tod_cos = np.cos(2 * np.pi * tod / SECONDS_IN_DAY)

    return doy_sin, doy_cos, tod_sin, tod_cos
