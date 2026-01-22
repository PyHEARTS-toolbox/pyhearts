## PyHEARTS HRV Metrics Reference (per-recording)

| Feature Name | Description | Calculation |
|---|---|---|
| average_heart_rate | Mean heart rate (bpm) | Compute RR→HR (60/(RR_ms/1000)) then mean; requires ≥60 RR intervals |
| sdnn | SD of NN (RR) intervals (ms) | std(RR_intervals_ms, ddof=1) |
| rmssd | Root mean square of successive RR differences (ms) | sqrt(mean(diff(RR_ms)^2)) |
| nn50 | Count of successive RR diffs > 50 ms | sum(abs(diff(RR_ms)) > 50) |
| pnn50 | Percent of successive RR diffs > 50 ms (%) | 100 * nn50 / (N-1) |
| sd1 | Poincaré short-term HRV (ms) | rmssd / sqrt(2) |
| sd2 | Poincaré long-term HRV (ms) | sqrt(2*sdnn^2 - sd1^2) |
