## PyHEARTS Beat-to-Beat Variability Metrics Reference (per-recording)

| Feature Name | Description | Calculation |
|---|---|---|
| QT_interval_ms_std | Standard deviation across cycles for QT_interval_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| QT_interval_ms_cv | Coefficient of variation across cycles (unitless) for QT_interval_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| QT_interval_ms_iqr | Interquartile range across cycles for QT_interval_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| QT_interval_ms_mad | Median absolute deviation across cycles (robust) for QT_interval_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| QT_interval_ms_range | Range across cycles for QT_interval_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| QRS_interval_ms_std | Standard deviation across cycles for QRS_interval_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| QRS_interval_ms_cv | Coefficient of variation across cycles (unitless) for QRS_interval_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| QRS_interval_ms_iqr | Interquartile range across cycles for QRS_interval_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| QRS_interval_ms_mad | Median absolute deviation across cycles (robust) for QRS_interval_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| QRS_interval_ms_range | Range across cycles for QRS_interval_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| PR_interval_ms_std | Standard deviation across cycles for PR_interval_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| PR_interval_ms_cv | Coefficient of variation across cycles (unitless) for PR_interval_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| PR_interval_ms_iqr | Interquartile range across cycles for PR_interval_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| PR_interval_ms_mad | Median absolute deviation across cycles (robust) for PR_interval_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| PR_interval_ms_range | Range across cycles for PR_interval_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| RR_interval_ms_std | Standard deviation across cycles for RR_interval_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| RR_interval_ms_cv | Coefficient of variation across cycles (unitless) for RR_interval_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| RR_interval_ms_iqr | Interquartile range across cycles for RR_interval_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| RR_interval_ms_mad | Median absolute deviation across cycles (robust) for RR_interval_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| RR_interval_ms_range | Range across cycles for RR_interval_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| QTc_Bazett_ms_std | Standard deviation across cycles for QTc_Bazett_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| QTc_Bazett_ms_cv | Coefficient of variation across cycles (unitless) for QTc_Bazett_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| QTc_Bazett_ms_iqr | Interquartile range across cycles for QTc_Bazett_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| QTc_Bazett_ms_mad | Median absolute deviation across cycles (robust) for QTc_Bazett_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| QTc_Bazett_ms_range | Range across cycles for QTc_Bazett_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| QTc_Fridericia_ms_std | Standard deviation across cycles for QTc_Fridericia_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| QTc_Fridericia_ms_cv | Coefficient of variation across cycles (unitless) for QTc_Fridericia_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| QTc_Fridericia_ms_iqr | Interquartile range across cycles for QTc_Fridericia_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| QTc_Fridericia_ms_mad | Median absolute deviation across cycles (robust) for QTc_Fridericia_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| QTc_Fridericia_ms_range | Range across cycles for QTc_Fridericia_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| R_gauss_height_std | Standard deviation across cycles for R_gauss_height | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| R_gauss_height_cv | Coefficient of variation across cycles (unitless) for R_gauss_height | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| R_gauss_height_iqr | Interquartile range across cycles for R_gauss_height | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| R_gauss_height_mad | Median absolute deviation across cycles (robust) for R_gauss_height | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| R_gauss_height_range | Range across cycles for R_gauss_height | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| P_gauss_height_std | Standard deviation across cycles for P_gauss_height | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| P_gauss_height_cv | Coefficient of variation across cycles (unitless) for P_gauss_height | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| P_gauss_height_iqr | Interquartile range across cycles for P_gauss_height | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| P_gauss_height_mad | Median absolute deviation across cycles (robust) for P_gauss_height | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| P_gauss_height_range | Range across cycles for P_gauss_height | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| T_gauss_height_std | Standard deviation across cycles for T_gauss_height | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| T_gauss_height_cv | Coefficient of variation across cycles (unitless) for T_gauss_height | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| T_gauss_height_iqr | Interquartile range across cycles for T_gauss_height | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| T_gauss_height_mad | Median absolute deviation across cycles (robust) for T_gauss_height | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| T_gauss_height_range | Range across cycles for T_gauss_height | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| R_duration_ms_std | Standard deviation across cycles for R_duration_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| R_duration_ms_cv | Coefficient of variation across cycles (unitless) for R_duration_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| R_duration_ms_iqr | Interquartile range across cycles for R_duration_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| R_duration_ms_mad | Median absolute deviation across cycles (robust) for R_duration_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| R_duration_ms_range | Range across cycles for R_duration_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| P_duration_ms_std | Standard deviation across cycles for P_duration_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| P_duration_ms_cv | Coefficient of variation across cycles (unitless) for P_duration_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| P_duration_ms_iqr | Interquartile range across cycles for P_duration_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| P_duration_ms_mad | Median absolute deviation across cycles (robust) for P_duration_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| P_duration_ms_range | Range across cycles for P_duration_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| T_duration_ms_std | Standard deviation across cycles for T_duration_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| T_duration_ms_cv | Coefficient of variation across cycles (unitless) for T_duration_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| T_duration_ms_iqr | Interquartile range across cycles for T_duration_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| T_duration_ms_mad | Median absolute deviation across cycles (robust) for T_duration_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| T_duration_ms_range | Range across cycles for T_duration_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
| ST_segment_ms_std | Standard deviation across cycles for ST_segment_ms | std(x, ddof=1) computed on non-NaN values; requires ≥2 values |
| ST_segment_ms_cv | Coefficient of variation across cycles (unitless) for ST_segment_ms | std(x, ddof=1) / abs(mean(x)) if abs(mean)>1e-6 else NaN; requires ≥2 values |
| ST_segment_ms_iqr | Interquartile range across cycles for ST_segment_ms | percentile(x,75) - percentile(x,25) computed on non-NaN values; requires ≥2 values |
| ST_segment_ms_mad | Median absolute deviation across cycles (robust) for ST_segment_ms | 1.4826 * median(abs(x - median(x))) computed on non-NaN values; requires ≥2 values |
| ST_segment_ms_range | Range across cycles for ST_segment_ms | max(x) - min(x) computed on non-NaN values; requires ≥2 values |
