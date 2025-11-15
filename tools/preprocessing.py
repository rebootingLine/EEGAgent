def preprocessing(raw, config):
    fs, l_freq, h_freq, notch_freq = config['fs'], config['l_freq'], config['h_freq'], config['notch_freq']
    raw.filter(l_freq=l_freq, h_freq=h_freq) 
    if notch_freq is not None: 
        raw.notch_filter(freqs=notch_freq)
    if raw.info['sfreq'] != fs:
        raw = raw.resample(sfreq=fs, npad='auto')
    return raw

