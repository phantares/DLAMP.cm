def scale_z(stats, data, variable_name):
    stat = stats.get(variable_name, None)

    if stat is not None:
        mean = stat["mean"]
        std = stat["std"]
        if std != 0:
            data = (data - mean) / std

    return data
