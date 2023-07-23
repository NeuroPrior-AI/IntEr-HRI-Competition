from scipy.signal import butter, cheby1, ellip, sosfilt
import pywt
import mne

class Filter:
    def __init__(self, raw, lowcut=0.1, highcut=30, order=5, rp=5, rs=40):
        self.raw = raw
        self.lowcut = lowcut
        self.highcut = highcut
        self.sfreq = raw.info['sfreq']
        self.order = order
        self.rp = rp
        self.rs = rs

    def filter_data(self, filter_type='cheby'):
        if filter_type == 'bandpass':
            self.raw.filter(l_freq=self.lowcut, h_freq=self.highcut)
            return self.raw
        elif filter_type in ['butter', 'cheby', 'ellip']:
            data = self.raw.get_data()
            data_filtered = self._bandpass_filter(data, filter_type)
            raw_filtered = self.raw.copy()
            raw_filtered._data = data_filtered
            return raw_filtered
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    # define filter, choose from [butter, cheby, ellip]
    def _bandpass(self, lowcut, highcut, fs, order, rp=None, rs=None, filter_type='butter'):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if filter_type == 'butter':
            sos = butter(order, [low, high], analog=False,
                         btype='band', output='sos')
        elif filter_type == 'cheby':
            sos = cheby1(order, rp, [low, high], btype='band', output='sos')
        elif filter_type == 'ellip':
            sos = ellip(order, rp, rs, [low, high], btype='band', output='sos')
        return sos

    def _bandpass_filter(self, data, filter_type='butter'):
        sos = self._bandpass(self.lowcut, self.highcut, self.sfreq,
                             self.order, self.rp, self.rs, filter_type)
        y = sosfilt(sos, data)
        return y

    def _wavelet_transform(self, data, wavelet='db4'):
        (cA, cD) = pywt.dwt(data, wavelet)
        return cA, cD
