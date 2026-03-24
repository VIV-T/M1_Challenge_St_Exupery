## **1. Dataset Overview**

* **Number of rows:** 364,545
* **Number of columns:** 195
* **Numeric columns:** 81
* **Categorical columns:** 86
* **Datetime columns:** 28
* **Total missing cells:** 29,428,590 (~41.4%)
* **Duplicate rows:** 0 (0%)

**Observations:**

* The dataset is **large and heterogeneous**, spanning numeric, categorical, and datetime fields.
* High missingness (~41%) is concentrated in operational, flight scheduling, and passenger flow metrics.
* No duplicate rows, ensuring unique records — valuable for model reliability.

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
* Features with 70–90% missing can be used cautiously with imputation or as missingness indicators.

---

## **3. Outlier Analysis**

Columns with extreme values or unusual entries:

* **Passenger counts:** `InvoiceNbPaxConnecting`, `FarmsNbPaxPHMR` have extreme high values.
* **Operational timings:** `TurnsBlockTimeMinutes`, `DeskDurationMinutes` contain negative or extremely high values.
* **Delays:** `DelayMainReasonDuration`, `DelayHBLDurationMinutes` exhibit extreme values.
* **PAX scanning:** `PxAvgTimeBetweenCheckingPIF`, `PxAvgTimeInTerminal` include negative and extremely high values.

**Implications:**

* Numerical columns should be **cleaned, capped, or transformed**.
* Negative dwell or interval times likely indicate **data entry errors or encoded missing values**.

---

## **4. Correlation Highlights**

High correlations detected between operational and passenger metrics:

| Column 1                | Column 2                    | Correlation |
| ----------------------- | --------------------------- | ----------- |
| `InvoiceNbPaxHTransit`  | `InvoiceNbPaxTotal`         | 0.997       |
| `TurnsBlockTimeMinutes` | `InvoicePkgDurationMinutes` | 0.996       |
| `PxScansPIF`            | `PxScansAccPIF`             | 0.992       |
| `NbPax`                 | `NbPaxHTransit`             | 0.992       |

**Implications:**

* High correlation → consider **dimensionality reduction**, e.g., PCA or feature selection.

---

## **5. Datetime Coverage**

* Data spans **2020-03-20 → 2027-06-02** across multiple datetime types.
* Flight and passenger events concentrated in **2023–2026**.
* Some columns, like `LTCtGLastCallDatetime`, are completely empty.

**Implications:**

* Time-based features are rich but require **UTC alignment, lag features, and careful handling of missing timestamps**.

---

## **6. Feature Engineering Flags**

| Flag                     | Count / Examples                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| High missing (>30%)      | `IdDelayMainReasonSubcode`, `Counter`, `Conveyor`, `LTCancellationDatetime`, PIF metrics         |
| Low variance (≤2 unique) | `NbFlight`, `AirportCode`, `IdTerminalType`                                                      |
| High cardinality (>100)  | `IdMovementVinci`, `IdFarms`, `IdAircraftType`, `airlineOACICode`, `SysStopover`, `flightNumber` |

**Implications:**

* High missing → consider imputation or dropping.
* Low variance → likely non-informative; candidate for removal.
* High cardinality → requires encoding strategies (target encoding, embeddings, hashing).

---

## **7. Categorical Data Insights**

* Sparse categorical variables (e.g., `BusCauseModif`, `ResponsableModifCtG`) have high missing or very few unique values.
* Frequent categories exist in `DelayMainReason` and `DelayMainReasonSubcode`.

**Implications:**

* Missingness itself can be a **predictive signal**.
* Rare categories → group under “Other” or apply **frequency encoding**.

---

## **8. Numeric Data Observations**

* Passenger counts, block times, and dwell times exhibit **extreme skew and outliers**.
* PAX scanning intervals sometimes have **negative values**, needing cleaning or capping.
* Some operational durations are **orders of magnitude higher than typical values**, likely data entry errors.

---

## **9. Recommended Next Steps Before Modeling**

1. **Data Cleaning**

   * Handle negative and extreme numeric values.
   * Decide strategy for >90% missing columns (drop or impute).

2. **Feature Engineering**

   * Encode high-cardinality categorical variables.
   * Derive features from timestamps (flight duration, day-of-week, peak/off-peak).
   * Aggregate or reduce highly correlated columns.

3. **Leakage Detection**

   * Ensure target variables are **not derived from future information**.

4. **Outlier Handling**

   * Apply winsorization or robust scaling for operational metrics and passenger counts.

5. **Modeling Strategy**

   * Mixed data types → tree-based models (XGBoost, LightGBM) or neural networks with embeddings.
   * Missing values → consider missing indicator features.

---
