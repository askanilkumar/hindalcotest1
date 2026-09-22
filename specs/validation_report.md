AC1 - PASS - `python src/generate_data.py` completed successfully and generated exactly 2,000 records in `data/machine_data.csv`; CSV verification reported `ROWS=2000`.
AC2 - PASS - CSV verification reported the required columns: `temperature`, `vibration`, `pressure`, `operating_hours`, `load_percentage`, and `machine_failure`.
AC3 - PASS - `python src/train.py` completed without errors and printed the evaluation results.
AC4 - PASS - Training created `models/machine_failure_model.pkl`; artifact existence verification reported `MODEL_EXISTS=True`.
AC5 - PASS - Training displayed Accuracy `0.9750`, Precision `0.0000`, Recall `0.0000`, F1 Score `0.0000`, and confusion matrix `[[390, 0], [10, 0]]`.
AC6 - PASS - `python src/predict.py` completed successfully and returned `Predicted class: 0` and `Failure probability: 0.0000`.
AC7 - PASS - `python -m pytest tests -v` completed successfully: 3 tests passed in 6.54 seconds.
AC8 - PASS - `README.md` contains dependency installation, pipeline execution, prediction, pytest, output locations, and project-structure instructions.