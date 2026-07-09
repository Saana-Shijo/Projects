import numpy as np

# -----------------------------
# Time-domain Features
# -----------------------------

def energy(frame):
    return np.sum(frame ** 2)


def rms(frame):
    return np.sqrt(np.mean(frame ** 2))


def zcr(frame):
    return np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))


def peak(frame):
    return np.max(np.abs(frame))


def variance(frame):
    return np.var(frame)


# -----------------------------
# Frequency-domain Features
# -----------------------------

def spectral_centroid(frame, sr):
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1 / sr)

    if np.sum(spectrum) == 0:
        return 0

    return np.sum(freqs * spectrum) / np.sum(spectrum)


def spectral_flatness(frame):
    spectrum = np.abs(np.fft.rfft(frame)) + 1e-12

    geometric = np.exp(np.mean(np.log(spectrum)))
    arithmetic = np.mean(spectrum)

    return geometric / arithmetic


def spectral_entropy(frame):
    spectrum = np.abs(np.fft.rfft(frame))

    psd = spectrum ** 2
    psd_sum = np.sum(psd)

    if psd_sum == 0:
        return 0

    psd = psd / psd_sum

    return -np.sum(psd * np.log2(psd + 1e-12))


# -----------------------------
# Feature Extraction
# -----------------------------

def extract_features(frame, sr):

    return [

        energy(frame),

        rms(frame),

        zcr(frame),

        peak(frame),

        variance(frame),

        spectral_centroid(frame, sr),

        spectral_flatness(frame),

        spectral_entropy(frame)

    ]