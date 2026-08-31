Heart Disease Prediction with Ensemble Models
Overview
This project looks at how different preprocessing choices affect the performance of ensemble models on a heart disease dataset.
Instead of just training a model and reporting accuracy, the idea here was to run a few controlled experiments and see what actually changes when you handle outliers or missing data differently. The setup also uses pipelines to make sure everything is done properly (no data leakage).

Two models were tested:
* Random Forest
* XGBoost

Research Question :
Do preprocessing steps like outlier handling or imputation actually make a difference?

And if they do:

* Which methods help?
* Which ones don’t really matter?

How the experiment was set up
Data split
* 80/20 train-test split
* Stratified so class balance is preserved

Preprocessing methods tested
* Baseline (just basic imputation)
* Winsorization (to cap extreme values)
* Mean imputation
* KNN imputation

All of this was done using `scikit-learn` pipelines so that preprocessing happens inside training and cross-validation to avoids leakage and gives more reliable results.

Evaluation
* Metric used: **F1-score**
* Validation: **Stratified K-Fold Cross-Validation**
* Multiple random seeds to reduce randomness

Results
codes are not in modules, as you run the code, you get the result

What I observed with a thusand dataset
A few things stood out after running the experiments:
Winsorization
It helped a bit in some cases, especially for Random Forest. That makes sense since reducing extreme values can stabilize tree splits.
For XGBoost, the impact wasn’t that obvious. It already handles irregularities quite well, so the extra step didn’t change much.
Missing value handling
**Mean imputation** was simple but consistent
**KNN imputation** sometimes helped, but results were a bit less stable
So while KNN is more “intelligent”, it doesn’t always guarantee better performance.

Model behavior
* XGBoost was generally more stable across different setups
* Random Forest reacted more to preprocessing changes

Cross-validation vs test results
The CV scores and test results were fairly consistent, which is a good sign that the setup is working as intended.

Model explanation
SHAP was used to understand what the model is doing.
This helps show which features the model relies on most.

Reproducibility
If modules are created, it is reproducible for these reasons:
* Fixed random seed (`42`)
* Pipelines used for all preprocessing
* Results saved to files

Limitations
There are still a few gaps:
* No hyperparameter tuning (everything is mostly default)
* No statistical tests to compare models formally
* Only one dataset used

Next steps
If I were to extend this, I’d look into:
* Hyperparameter tuning (GridSearch / Optuna)
* Running the same setup on other datasets
* Adding statistical tests to compare results properly

Final thoughts
The main takeaway for me is that preprocessing does matter, but not always in obvious ways. Some methods help in certain situations, while others don’t make much difference.
More importantly, setting things up properly (especially avoiding data leakage) is just as important as the model itself.

Contributions
If you have suggestions or want to build on this, feel free to open an issue or PR.
