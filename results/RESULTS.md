# Results

31,999 headlines after de-duplication (50.0% clickbait), split 25,599 / 3,200 / 3,200 (train / val / test, stratified).

## Test set, default threshold 0.5

| Model | Accuracy | F1 | ROC AUC | PR AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| tfidf_logreg | 0.9859 | 0.9859 | 0.9982 | 0.9984 | 0.0120 | 0.0063 |
| textcnn_uncalibrated | 0.9694 | 0.9693 | 0.9957 | 0.9956 | 0.0222 | 0.0099 |
| textcnn_calibrated | 0.9694 | 0.9693 | 0.9957 | 0.9956 | 0.0224 | 0.0094 |

## Test set, threshold chosen on validation for 98% precision

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| tfidf_logreg | 0.393 | 0.9862 | 0.9831 | 0.9847 |
| textcnn_uncalibrated | 0.804 | 0.9894 | 0.9369 | 0.9624 |
| textcnn_calibrated | 0.750 | 0.9894 | 0.9369 | 0.9624 |

## Casing shortcut check

A model trained on original casing scores F1 0.9953 on original casing but 0.0198 when the same test headlines are lowercased. The gap is how much it leaned on publisher style (Title Case) rather than wording, which is why the pipeline lowercases everything.

TextCNN temperature fitted on validation: 1.282

## Strongest n-grams (logistic regression)

Clickbait: `you`, `" `, ` "`, ` 1`, ` a `, `'s `, `17`, `'s`, `these`, `21`, `yo`, ` you`

Not clickbait: `? `, `, `, `wins`, `may`, `says`, `mets`, `knicks`, `at`, `in`, `uk`, `obama`, `. `

![curves](curves.png)

See [errors.md](errors.md) for misclassified headlines.
