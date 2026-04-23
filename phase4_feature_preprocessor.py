import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class WESADFeaturePreprocessor:
    """Preprocess the EDA feature matrix produced in Phase 3.

    Phase 4 pipeline (applied in order):
    1. Impute NaN values (median per feature column).
    2. Per-subject z-score on EDA amplitude features to remove inter-subject
       baseline differences before pooling subjects together.
    3. Global StandardScaler on all feature columns for ML readiness.
    4. Save the preprocessed parquet and the fitted scaler.

    EDA amplitude features are those whose absolute scale is tied to
    the subject's skin conductance baseline (µS units).  All other
    features (counts, timing, spectral) are treated as global-only.
    """

    # Columns that carry raw EDA amplitude and are z-scored per subject first.
    AMPLITUDE_FEATURES: list[str] = [
        "mu_tonic",
        "sigma_tonic",
        "mu_ampl",
        "max_ampl",
        "eda_std",
        "eda_mean",
        "eda_median",
        "eda_p25",
        "eda_p75",
    ]

    # All numeric feature columns (amplitude + non-amplitude).
    ALL_FEATURES: list[str] = [
        "mu_tonic",
        "slope_tonic",
        "sigma_tonic",
        "nscr",
        "mu_ampl",
        "max_ampl",
        "rise_time",
        "decay_time",
        "eda_std",
        "eda_mean",
        "eda_median",
        "eda_p25",
        "eda_p75",
        "power_0_02",
        "power_02_05",
    ]

    # Non-feature metadata columns preserved as-is.
    META_COLS: list[str] = ["subject", "window_id", "binary_label"]

    def __init__(self, input_parquet: str, output_dir: str = "output"):
        self.input_parquet = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_features(self) -> pd.DataFrame:
        self.df = pd.read_parquet(self.input_parquet)
        return self.df

    def save_preprocessed(
        self,
        parquet_filename: str = "wesad_features_phase4.parquet",
        scaler_filename: str = "wesad_scaler_phase4.pkl",
    ) -> tuple[pd.DataFrame, Path, Path]:
        processed_df = self.preprocess()

        parquet_out = self.output_dir / parquet_filename
        scaler_out = self.output_dir / scaler_filename

        processed_df.to_parquet(parquet_out, index=False)
        with open(scaler_out, "wb") as f:
            pickle.dump(self.scaler, f)

        return processed_df, parquet_out, scaler_out

    # ------------------------------------------------------------------
    # Preprocessing steps
    # ------------------------------------------------------------------

    def impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1 — Median imputation for each feature column."""
        df = df.copy()
        nan_counts = df[self.ALL_FEATURES].isna().sum()
        imputed = nan_counts[nan_counts > 0]
        if not imputed.empty:
            print("  [1] Imputing NaNs (median):")
            for col, n in imputed.items():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"      {col}: {n} NaN(s) → {median_val:.6f}")
        else:
            print("  [1] No NaN values found — skipping imputation.")
        return df

    def per_subject_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2 — Z-score amplitude features within each subject.

        Divides by std only when std > 0; leaves the column unchanged
        (after mean centering) if the subject has zero variance in that
        feature (e.g. a single window).
        """
        df = df.copy()
        print(f"  [2] Per-subject z-score on {len(self.AMPLITUDE_FEATURES)} amplitude features.")

        def _zscore_group(grp: pd.DataFrame) -> pd.DataFrame:
            for col in self.AMPLITUDE_FEATURES:
                mean = grp[col].mean()
                std = grp[col].std(ddof=0)
                grp[col] = (grp[col] - mean) / std if std > 0 else grp[col] - mean
            return grp

        df[self.AMPLITUDE_FEATURES] = (
            df.groupby("subject", group_keys=False)[self.AMPLITUDE_FEATURES]
            .apply(_zscore_group)
        )
        return df

    def global_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3 — Fit a StandardScaler on all feature columns."""
        df = df.copy()
        print(f"  [3] Global StandardScaler on {len(self.ALL_FEATURES)} features.")
        self.scaler = StandardScaler()
        df[self.ALL_FEATURES] = self.scaler.fit_transform(df[self.ALL_FEATURES])
        return df

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def preprocess(self) -> pd.DataFrame:
        """Run the full preprocessing pipeline and return the result."""
        self.load_features()
        print(f"\nPreprocessing {len(self.df)} windows × {len(self.ALL_FEATURES)} features...")
        df = self.impute_nans(self.df)
        df = self.per_subject_zscore(df)
        df = self.global_scale(df)
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """Return mean/std/min/max of each feature after preprocessing."""
        stats = processed_df[self.ALL_FEATURES].agg(["mean", "std", "min", "max"]).T
        stats.columns = ["mean", "std", "min", "max"]
        return stats.round(4)

    def nan_report(self) -> pd.Series:
        """Return NaN counts in the raw loaded data."""
        if self.df is None:
            self.load_features()
        return self.df[self.ALL_FEATURES].isna().sum()


if __name__ == "__main__":
    preprocessor = WESADFeaturePreprocessor("output/wesad_eda_features_phase3.parquet")
    processed_df, parquet_out, scaler_out = preprocessor.save_preprocessed()

    print(f"\n==========================================")
    print(f"Preprocessed DataFrame shape: {processed_df.shape}")
    print(f"Saved parquet to:  {parquet_out}")
    print(f"Saved scaler to:   {scaler_out}")
    print("\nFeature statistics after preprocessing:")
    print(preprocessor.summary(processed_df).to_string())
