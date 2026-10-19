# Verification of the maximum log-likelihoods of the BIC tables

670 of 670 cells verified.

Reproduction: |reproduced - stored| log-likelihood above 1e-3 in 66 cells; stored sample outside the prior box in 2 cells.

Refinement: largest increase of the log-likelihood over the reproduced value 2630.560; cells improved by more than 1 unit: 106.

## Table S1.2
### control_sensitive
- global lowest BIC: Logistic (weight 0.952) before, Logistic (weight 0.954) after
- mice whose lowest-BIC model changes: 0 of 9
  - Global von Bertalanffy: log L -1141.42 -> -1140.42
  - Global Linear: log L -1371.22 -> -1368.51

### control_resistant
- global lowest BIC: Logistic (weight 1.000) before, Logistic (weight 1.000) after
- mice whose lowest-BIC model changes: 0 of 5

## Table 4
### control
- lowest BIC: Scenario 2 before, Scenario 2 after

## Table S2.2
### radiation_sensitive
- global lowest BIC: CDEF (weight 0.809) before, CDEF (weight 0.829) after
- mice whose lowest-BIC model changes: 2 of 15 (c5_m18: AED -> ACD; c5_m26: AED -> ED)
  - c0_m16 ACD: log L -161.94 -> -139.57
  - c0_m8 ACD: log L -157.00 -> -128.54
  - c5_m18 ACD: log L -129.62 -> -128.59
  - c5_m25 ACD: log L -143.49 -> -123.64
  - Global ACD: log L -2129.17 -> -2127.91
  - c0_m16 AED: log L -161.19 -> -139.58
  - c0_m23 AED: log L -123.87 -> -110.16
  - Global AED: log L -2133.03 -> -2131.92
  - Global CDEF: log L -2120.37 -> -2118.91
  - Global CDLF: log L -2126.08 -> -2124.98
  - Global EDEF: log L -2121.84 -> -2120.51
  - Global EDLF: log L -2130.57 -> -2129.54
  - Global EDSF: log L -2142.39 -> -2141.17
  - Global ED: log L -2139.10 -> -2137.81
  - Global LQ: log L -2319.52 -> -2318.05

### radiation_resistant
- global lowest BIC: ACD (weight 0.412) before, ACD (weight 0.451) after
- mice whose lowest-BIC model changes: 3 of 9 (c1_m1: EDSF -> CDSF; c1_m3: AED -> ED; c1_m6: CD -> ACD)
  - c1_m9 ACD: log L -128.92 -> -126.87
  - c1_m9 AED: log L -130.25 -> -126.88

## Table S2.3
### radiation_sensitive
- global lowest BIC: CD w/ delay (weight 0.527) before, ED w/ delay (weight 0.485) after  **(changed)**
- mice whose lowest-BIC model changes: 4 of 15 (c0_m23: EDSF w/ delay -> CDSF w/ delay; c5_m12: EDEF w/ delay -> CDSF w/ delay; c5_m15: EDSF w/ delay -> CDSF w/ delay; c5_m6: AED w/ delay -> EDSF w/ delay)
  - c0_m16 ACD w/ delay: log L -159.79 -> -139.57
  - c0_m8 ACD w/ delay: log L -149.04 -> -125.40
  - c5_m12 ACD w/ delay: log L -127.47 -> -113.01
  - c5_m15 ACD w/ delay: log L -124.28 -> -118.53
  - c5_m18 ACD w/ delay: log L -127.61 -> -126.52
  - c5_m21 ACD w/ delay: log L -158.58 -> -148.97
  - c5_m25 ACD w/ delay: log L -144.39 -> -120.89
  - c5_m3 ACD w/ delay: log L -112.46 -> -108.05
  - Global ACD w/ delay: log L -2121.08 -> -2119.94
  - c0_m16 AED w/ delay: log L -159.98 -> -139.58
  - c0_m8 AED w/ delay: log L -148.82 -> -125.32
  - c5_m12 AED w/ delay: log L -124.99 -> -113.32
  - c5_m15 AED w/ delay: log L -124.33 -> -118.56
  - c5_m25 AED w/ delay: log L -138.72 -> -121.03
  - c5_m3 AED w/ delay: log L -112.16 -> -105.62
  - Global AED w/ delay: log L -2121.89 -> -2120.30
  - c0_m23 CDEF w/ delay: log L -102.18 -> -99.95
  - c0_m8 CDEF w/ delay: log L -129.16 -> -125.00
  - c5_m12 CDEF w/ delay: log L -129.65 -> -112.68
  - c5_m15 CDEF w/ delay: log L -137.06 -> -117.23
  - c5_m18 CDEF w/ delay: log L -132.54 -> -127.80
  - c5_m19 CDEF w/ delay: log L -131.04 -> -119.67
  - c5_m21 CDEF w/ delay: log L -158.21 -> -147.86
  - c5_m22 CDEF w/ delay: log L -120.97 -> -116.31
  - c5_m25 CDEF w/ delay: log L -173.12 -> -120.33
  - c5_m4 CDEF w/ delay: log L -127.66 -> -115.96
  - c5_m5 CDEF w/ delay: log L -127.05 -> -114.89
  - Global CDEF w/ delay: log L -2119.08 -> -2116.07
  - c0_m23 CDSF w/ delay: log L -99.45 -> -92.12
  - c5_m6 CDSF w/ delay: log L -141.49 -> -136.97
  - Global CDSF w/ delay: log L -2115.62 -> -2114.21
  - c5_m3 CD w/ delay: log L -109.10 -> -108.05
  - Global CD w/ delay: log L -2115.31 -> -2114.21
  - c0_m16 EDEF w/ delay: log L -151.90 -> -138.58
  - c0_m23 EDEF w/ delay: log L -104.05 -> -101.32
  - c0_m8 EDEF w/ delay: log L -133.20 -> -125.20
  - c5_m12 EDEF w/ delay: log L -1418.59 -> -113.81
  - c5_m15 EDEF w/ delay: log L -141.62 -> -117.33
  - c5_m18 EDEF w/ delay: log L -143.40 -> -128.09
  - c5_m19 EDEF w/ delay: log L -258.00 -> -120.95
  - c5_m21 EDEF w/ delay: log L -202.20 -> -148.25
  - c5_m22 EDEF w/ delay: log L -202.48 -> -116.78
  - c5_m25 EDEF w/ delay: log L -191.56 -> -120.89
  - c5_m4 EDEF w/ delay: log L -118.34 -> -116.94
  - c5_m5 EDEF w/ delay: log L -127.43 -> -115.93
  - Global EDEF w/ delay: log L -2444.24 -> -2118.08
  - c5_m6 EDSF w/ delay: log L -138.98 -> -136.54
  - Global EDSF w/ delay: log L -2115.32 -> -2113.76
  - Global ED w/ delay: log L -2115.69 -> -2114.13

### radiation_resistant
- global lowest BIC: CD w/ delay (weight 0.260) before, ED w/ delay (weight 0.258) after  **(changed)**
- mice whose lowest-BIC model changes: 5 of 9 (c1_m10: CDEF w/ delay -> AED w/ delay; c1_m11: ED w/ delay -> AED w/ delay; c1_m13: CD w/ delay -> ACD w/ delay; c1_m3: ED w/ delay -> AED w/ delay; c1_m9: ED w/ delay -> CDSF w/ delay)
  - c1_m9 ACD w/ delay: log L -129.57 -> -126.85
  - c1_m9 AED w/ delay: log L -130.45 -> -126.85
  - c1_m10 CDEF w/ delay: log L -428.64 -> -47.40
  - c1_m11 CDEF w/ delay: log L -123.97 -> -116.44
  - c1_m13 CDEF w/ delay: log L -49.52 -> -41.82
  - c1_m14 CDEF w/ delay: log L -181.66 -> -129.96
  - c1_m1 CDEF w/ delay: log L -534.50 -> -99.50
  - c1_m8 CDEF w/ delay: log L -696.37 -> -112.00
  - Global CDEF w/ delay: log L -827.92 -> -821.49
  - c1_m9 CDSF w/ delay: log L -126.29 -> -124.64
  - c1_m10 EDEF w/ delay: log L -630.47 -> -47.62
  - c1_m11 EDEF w/ delay: log L -135.98 -> -116.37
  - c1_m13 EDEF w/ delay: log L -86.27 -> -42.36
  - c1_m14 EDEF w/ delay: log L -242.79 -> -129.98
  - c1_m1 EDEF w/ delay: log L -714.84 -> -99.70
  - c1_m3 EDEF w/ delay: log L -33.74 -> -31.05
  - c1_m6 EDEF w/ delay: log L -48.40 -> -35.40
  - c1_m8 EDEF w/ delay: log L -2742.62 -> -112.06
  - Global EDEF w/ delay: log L -1509.69 -> -821.87
  - c1_m9 EDSF w/ delay: log L -126.81 -> -124.85

## Table 5
### radiation
- lowest BIC: Scenario 2 before, Scenario 2 after
  - V1 Scenario 1: log L -2938.65 -> -2935.61
  - V2 Scenario 2: log L -2939.50 -> -2936.75
  - V3 Scenario 3: log L -2957.68 -> -2954.65
  - V4 Scenario 4: log L -3170.75 -> -3166.87
  - V5 Scenario 5: log L -2939.17 -> -2935.64
  - V6 Scenario 6: log L -2954.76 -> -2951.15
  - V7 Scenario 7: log L -2939.26 -> -2936.08

## Table S3.2
### io_sensitive
- global lowest BIC: PR (weight 1.000) before, PR (weight 1.000) after
- mice whose lowest-BIC model changes: 3 of 3 (c0_m28: ACD -> MM; c0_m3: ACD -> MM; c0_m35: ACD -> CD)
  - c0_m28 ACD: log L -358.56 -> -113.71
  - c0_m35 ACD: log L -195.24 -> -150.15
  - c0_m3 ACD: log L -101.62 -> -86.16
  - Global ACD: log L -953.39 -> -407.59
  - Global CDEF: log L -408.29 -> -403.88
  - c0_m35 CD: log L -156.02 -> -150.14
  - Global CD: log L -411.14 -> -407.60
  - Global EDEF: log L -403.17 -> -401.50
  - Global EDSF: log L -401.28 -> -398.12

## Table SUV
### rad+io_sensitive
- global lowest BIC: Linear (weight 1.000) before, Linear (weight 1.000) after
  - Global Linear: log L -2203.87 -> -2202.09
  - Global Saturating exponential: log L -2213.15 -> -2210.56

