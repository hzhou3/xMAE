| Index | Feature              | Category   | Description                                                                 |
|-------|----------------------|------------|-----------------------------------------------------------------------------|
| 1     | sig_mean             | Amplitude  | Mean value of the PPG signal amplitude                                       |
| 2     | sig_std              | Amplitude  | Standard deviation of signal amplitude                                       |
| 3     | sig_median           | Amplitude  | Median amplitude of the signal                                               |
| 4     | sig_q25              | Amplitude  | 25th percentile of signal amplitude                                          |
| 5     | sig_q75              | Amplitude  | 75th percentile of signal amplitude                                          |
| 6     | sig_iqr              | Amplitude  | Interquartile range (q75 - q25)                                              |
| 7     | sig_min              | Amplitude  | Minimum signal amplitude                                                     |
| 8     | sig_max              | Amplitude  | Maximum signal amplitude                                                     |
| 9     | sig_amp_range        | Amplitude  | Range of amplitude (max - min)                                               |
| 10    | sig_rms              | Amplitude  | Root mean square of signal                                                   |
| 11    | sig_mad              | Amplitude  | Median absolute deviation                                                    |
| 12    | sig_skew             | Morphology | Skewness (asymmetry of waveform shape)                                       |
| 13    | sig_kurtosis         | Morphology | Kurtosis (peakedness / tail heaviness)                                       |
| 14    | diff1_std            | Derivative | Std of first derivative (slope variability)                                  |
| 15    | diff1_abs_mean       | Derivative | Mean absolute first derivative                                               |
| 16    | diff2_std            | Derivative | Std of second derivative (curvature variability)                             |
| 17    | zero_cross_rate      | Morphology | Rate of zero crossings (oscillation / waveform frequency proxy)              |
| 18    | peak_amp_mean        | Morphology | Mean amplitude of detected peaks                                             |
| 19    | peak_amp_std         | Morphology | Std of peak amplitudes                                                       |
| 20    | peak_amp_max         | Morphology | Maximum peak amplitude                                                       |
| 21    | n_peaks              | Timing     | Number of detected peaks (beats)                                             |
| 22    | peak_rate_per_sec    | Timing     | Peak rate per second                                                         |
| 23    | ibi_mean             | Timing     | Mean inter-beat interval                                                     |
| 24    | ibi_median           | Timing     | Median inter-beat interval                                                   |
| 25    | ibi_std              | Timing     | Std of inter-beat intervals                                                  |
| 26    | ibi_min              | Timing     | Minimum inter-beat interval                                                  |
| 27    | ibi_max              | Timing     | Maximum inter-beat interval                                                  |
| 28    | hr_mean              | Timing     | Mean heart rate                                                             |
| 29    | hr_median            | Timing     | Median heart rate                                                           |
| 30    | hr_std               | Timing     | Std of heart rate                                                           |
| 31    | hr_min               | Timing     | Minimum heart rate                                                          |
| 32    | hr_max               | Timing     | Maximum heart rate                                                          |
| 33    | rmssd                | Timing     | Short-term HRV (successive interval differences)                             |
| 34    | sdnn                 | Timing     | Overall HRV (std of intervals)                                               |
| 35    | pnn50                | Timing     | % of intervals differing >50 ms                                              |
