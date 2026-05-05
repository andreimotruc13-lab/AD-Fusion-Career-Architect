import sys
import joblib
import sklearn.compose._column_transformer as ct

# Program that updates the version of the pkl files in case they don't work
class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

try:
    recommender = joblib.load("recommender_fixed.pkl")
    preprocessor = joblib.load("preprocessor_fixed.pkl")
    print("✅ Files loaded successfully using the patch!")

    joblib.dump(recommender, "recommender_fixed.pkl")
    joblib.dump(preprocessor, "preprocessor_fixed.pkl")
    print("'recommender_fixed.pkl' and 'preprocessor_fixed.pkl' have been saved")

except Exception as e:
    print(f"Still fails: {e}")