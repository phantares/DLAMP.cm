from datetime import datetime, timedelta
import calendar
import numpy as np
import torch


SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
SECONDS_IN_DAY = HOURS_IN_DAY * MINUTES_IN_HOUR * SECONDS_IN_MINUTE


def encode_time(lon, utc_time: datetime, dtype=torch.float64):
    utc_tod = (
        utc_time.hour * MINUTES_IN_HOUR * SECONDS_IN_MINUTE
        + utc_time.minute * SECONDS_IN_MINUTE
        + utc_time.second
    )

    local_tod_seconds = utc_tod + lon * (SECONDS_IN_DAY / 360)
    local_tod = local_tod_seconds % SECONDS_IN_DAY

    tod_sin = np.sin(2 * np.pi * local_tod / SECONDS_IN_DAY)
    tod_cos = np.cos(2 * np.pi * local_tod / SECONDS_IN_DAY)

    day_offset = np.floor(local_tod_seconds / SECONDS_IN_DAY).astype(int)
    local_date = utc_time.date() + np.vectorize(lambda d: timedelta(days=int(d)))(
        day_offset
    )

    def get_doy_encoding(d):
        days_in_year = 366 if calendar.isleap(d.year) else 365
        doy = int(d.strftime("%j"))
        return (
            np.sin(2 * np.pi * doy / days_in_year),
            np.cos(2 * np.pi * doy / days_in_year),
        )

    doy_sin, doy_cos = np.vectorize(get_doy_encoding)(local_date)

    return torch.from_numpy(np.array([doy_sin, doy_cos, tod_sin, tod_cos])).to(dtype)
