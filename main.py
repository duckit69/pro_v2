from phase1_wesad_packager import WESADRawPackager
from phase2_wesad_window_builder import WESADPhase2WindowBuilder
from phase3_eda_feature_extractor import WESADEDAFeatureExtractor
from phase4_feature_preprocessor import WESADFeaturePreprocessor

# ------------------------------------------------------------------
# Phase 1 – Pack raw subject pickles into a single parquet
# ------------------------------------------------------------------
print("=" * 50)
print("Phase 1: Packaging raw WESAD data...")
print("=" * 50)
packager = WESADRawPackager(root_dir='data')
df1, out1 = packager.save_parquet()
print(packager.summary(df1).to_string(index=False))
print(f"Saved to: {out1}\n")

# ------------------------------------------------------------------
# Phase 2 – Slice into fixed EDA windows, assign binary labels
# ------------------------------------------------------------------
print("=" * 50)
print("Phase 2: Building EDA windows...")
print("=" * 50)
builder = WESADPhase2WindowBuilder(str(out1))
df2, out2 = builder.save_windows(window_seconds=60, step_seconds=30, eda_fs=4)
print(builder.summary(df2))
print(f"Saved to: {out2}\n")

# ------------------------------------------------------------------
# Phase 3 – Extract EDA features from every window
# ------------------------------------------------------------------
print("=" * 50)
print("Phase 3: Extracting EDA features...")
print("=" * 50)
extractor = WESADEDAFeatureExtractor(str(out2))
df3, out3 = extractor.save_features()
print(f"\nSuccessfully processed {len(df3)} total windows.")
print(f"Features DataFrame shape: {df3.shape}")
print(f"Saved to: {out3}\n")
print("Per-subject summary:")
print(extractor.summary(df3).to_string())

# ------------------------------------------------------------------
# Phase 4 – Preprocess / normalise features
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("Phase 4: Preprocessing & normalising features...")
print("=" * 50)
preprocessor = WESADFeaturePreprocessor(str(out3))
df4, out4, scaler_out = preprocessor.save_preprocessed()
print(f"\nPreprocessed DataFrame shape: {df4.shape}")
print(f"Saved parquet to: {out4}")
print(f"Saved scaler to:  {scaler_out}")
print("\nFeature statistics after preprocessing:")
print(preprocessor.summary(df4).to_string())