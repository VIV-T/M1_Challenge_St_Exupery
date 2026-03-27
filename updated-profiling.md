# **Full Profiling Report — Raw Flight Dataset**

**Date:** 2026-03-27
**Author:** Data Science Team

This report summarizes **profiling of the raw dataset**, including missingness, outliers, multicollinearity, feature importance, temporal coverage, and holiday/seasonal effects, with **recommendations for preprocessing and feature engineering**.

---

## **1. Dataset Overview**

* **Number of rows:** 364,545
* **Number of columns:** 195
* **Numeric columns:** 81
* **Categorical columns:** 86
* **Datetime columns:** 28
* **Total missing cells:** ~29,428,590 (~41.4%)
* **Duplicate rows:** 0 (0%)

**Observations:**

* The dataset is **large and heterogeneous**, spanning numeric, categorical, and datetime fields.
* High missingness (~41%) is concentrated in **operational, flight scheduling, and passenger flow metrics**.
* No duplicate rows, ensuring **unique records**, valuable for model reliability.

---

## **2. Missing Data Analysis**

Key columns with high missingness (>70%):

| Column                                                                            | Missing % | Notes                                        |
| --------------------------------------------------------------------------------- | --------- | -------------------------------------------- |
| `LTCtGLastCallDatetime`                                                           | 100%      | Completely empty                             |
| `InvoiceRevenue`, `InvoiceImputationDate`, `InvoiceDate`, `InvoiceNbOfTouchAndGo` | 100%      | Placeholder or incomplete invoice data       |
| `BusCauseModif`                                                                   | 98.3%     | Sparse operational metric                    |
| `BusRotationSchedule`, `BusRotationActual`                                        | 97%       | Sparse operational metrics                   |
| `LTCancellationDatetime`                                                          | 96.4%     | Most flights not cancelled                   |
| Desk operations (`LTDeskOpenScheduledDatetime`, `LTDeskCloseScheduledDatetime`)   | 73%       | Most flights don’t have desk scheduling info |
| Passenger flow & PIF metrics (`PxAvg*`, `PxScans*`)                               | 87–89%    | Sparse scanning metrics                      |

**Implications:**

* Columns with >90% missing are **not useful for modeling** unless imputation is feasible.
* Features with 70–90% missing can be used cautiously with **imputation or missingness indicators**.

---

## **3. Outlier Analysis**

Columns with extreme values or unusual entries:

* **Passenger counts:** `InvoiceNbPaxConnecting`, `FarmsNbPaxPHMR`, `NbPaxTransit` have extreme high values.
* **Operational timings:** `TurnsBlockTimeMinutes`, `DeskDurationMinutes` contain negative or extremely high values.
* **Delays:** `DelayMainReasonDuration`, `DelayHBLDurationMinutes` exhibit extreme values.
* **PAX scanning:** `PxAvgTimeBetweenCheckingPIF`, `PxAvgTimeInTerminal` include negative or extremely high values.

**Implications:**

* Numerical columns should be **cleaned, capped, or transformed**.
* Negative dwell or interval times likely indicate **data entry errors or encoded missing values**.

---

## **4. Correlation & Multicollinearity Highlights**

High correlations detected:

| Column 1                | Column 2                      | Correlation |
| ----------------------- | ----------------------------- | ----------- |
| `InvoiceNbPaxHTransit`  | `InvoiceNbPaxTotal`           | 0.997       |
| `TurnsBlockTimeMinutes` | `InvoicePkgDurationMinutes`   | 0.998       |
| `PxScansPIF`            | `PxScansAccPIF`               | 0.993       |
| `NbPax`                 | `NbPaxHTransit`               | 0.992       |
| `LTBagDeliveryDuration` | `PxAvgTimeBetweenCheckingPIF` | 1.0         |

**Implications:**

* High correlation → consider **dropping redundant columns** or using dimensionality reduction.
* Keep features with **high predictive power** per the feature score table.

---

## **5. Temporal Coverage**

* Data spans **2020-03-20 → 2026-10-25** across multiple datetime types.
* Flight and passenger events concentrated in **2023–2026**.
* Some columns, like `LTCtGLastCallDatetime`, are completely empty.

**Implications:**

* Temporal features are rich but require **UTC alignment, lag features, and careful handling of missing timestamps**.
* Avoid leakage when creating **rolling averages and historical features**.

---

## **6. Seasonality & Holidays**

| Season | Mean NbPaxTotal | Mean FarmsNbPaxPHMR |
| ------ | --------------- | ------------------- |
| Summer | 118.2           | 0.4605              |
| Fall   | 105.8           | 0.4507              |
| Spring | 104.4           | 0.4045              |
| Winter | 101.6           | 0.4808              |

| IsHoliday | Mean NbPaxTotal | PRM Load Factor |
| --------- | --------------- | --------------- |
| False     | 107.3           | 0.0027          |

**Observations:**

* Summer peaks passenger counts, winter is lowest.
* Holidays show minor impact on average but may spike for specific flights.

**Recommendations:**

* Encode **seasonal flags** and **holiday indicators**.
* Include interactions with **route and terminal**.

---

## **7. Feature Engineering Flags**

| Flag                     | Count / Examples                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| High missing (>30%)      | `IdDelayMainReasonSubcode`, `Counter`, `Conveyor`, `LTCancellationDatetime`, PIF metrics         |
| Low variance (≤2 unique) | `NbFlight`, `AirportCode`, `IdTerminalType`                                                      |
| High cardinality (>100)  | `IdMovementVinci`, `IdFarms`, `IdAircraftType`, `airlineOACICode`, `SysStopover`, `flightNumber` |

**Implications:**

* High missing → consider imputation or dropping.
* Low variance → likely non-informative.
* High cardinality → use **target encoding, hashing, or embeddings**.

---

## **8. Categorical Data Insights**

* Sparse categorical variables (e.g., `BusCauseModif`, `ResponsableModifCtG`) have high missing or few unique values.
* Frequent categories exist in `DelayMainReason` and `DelayMainReasonSubcode`.

**Recommendations:**

* Use **missingness as a feature**.
* Group rare categories under “Other” or **apply frequency/target encoding**.

---

## **9. Numeric Data Observations**

* Passenger counts, block times, and dwell times exhibit **skew and extreme values**.
* PAX scanning intervals sometimes have **negative values**.
* Some operational durations are likely **data entry errors**.

---

## **10. Recommended Next Steps Before Modeling**

1. **Data Cleaning**

   * Cap or transform extreme numerical values.
   * Decide strategy for >90% missing columns (drop or impute).
   * Handle negative dwell/interval times carefully.

2. **Feature Engineering**

   * Encode high-cardinality categorical variables.
   * Derive **temporal features**: hour-of-day, day-of-week, week-of-year, month, season, holiday flag.
   * Create **rolling historical features**: past 7/30-day average passengers per route/airline.
   * Reduce highly correlated features; retain most predictive.

3. **Leakage Detection**

   * Ensure **targets are not derived from future features**.
   * Align historical features strictly **before flight datetime**.

4. **Outlier Handling**

   * Winsorization or robust scaling for operational metrics and passenger counts.
   * Flag extreme values to preserve signal.

5. **Modeling Strategy**

   * Tree-based models (LightGBM, XGBoost) handle mixed data types and missing values.
   * Missingness itself can be used as a predictive feature.
   * PRM passengers → predict separately using `PRM_ratio = PRM / passengers`.

---
