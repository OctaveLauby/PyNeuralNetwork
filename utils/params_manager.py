def read_params(params, dft_params):
    """Return parameters completed with default.

    Raises:
        KeyError for unknown params
    """
    res_params = dft_params

    unknown_keys = []
    for (key, value) in params.items():
        if key not in res_params:
            unknown_keys.append(key)
        else:
            res_params[key] = value

    if unknown_keys:
        raise KeyError(
            "Unexpected keys in params: %s" % ", ".join(unknown_keys)
        )

    return res_params
