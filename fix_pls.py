import sys
import joblib
import sklearn.compose._column_transformer as ct

# 1. Manually inject the missing attribute into the scikit-learn module
class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

# 2. Load your original "broken" files
try:
    recommender = joblib.load("recommender.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    print("✅ Files loaded successfully using the patch!")

    # 3. Save them again using your CURRENT scikit-learn version
    # This 'bakes in' the new structure so you don't need the patch anymore
    joblib.dump(recommender, "recommender_fixed.pkl")
    joblib.dump(preprocessor, "preprocessor_fixed.pkl")
    print("🚀 Fixed files saved as 'recommender_fixed.pkl' and 'preprocessor_fixed.pkl'")

except Exception as e:
    print(f"❌ Still failing: {e}")