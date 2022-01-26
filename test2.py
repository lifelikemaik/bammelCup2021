import joblib
import pandas as pd

# load model predictions
loaded_predictions = joblib.load("./random_forest.joblib")
loaded_predictions_df = pd.DataFrame({'Column1': loaded_predictions[:, 0], 'Column2': loaded_predictions[:, 1]})

# load test results
test_res = pd.read_csv("submission_random.csv", names=['id', 'prediction'])






