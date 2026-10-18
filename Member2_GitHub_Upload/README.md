# Member 2 Final Delivery
## CLIP Encoder + Concept Prediction Layer

### 1. Task

Member 2 is responsible for:

1. Loading a pretrained CLIP image encoder.
2. Extracting image features from CUB-200-2011.
3. Training the Concept Prediction Layer / Concept Bottleneck Layer.
4. Producing concept scores for each image.
5. Saving the trained concept layer and prediction results for downstream use.

---

## 2. Dataset

Dataset: CUB-200-2011

Total images: 11,788

Experiment split:

- Train: 5,394 images
- Validation: 600 images
- Test: 5,794 images

Number of bird classes: 200

Number of concepts / attributes: 312

Concept order follows:

attribute_1 ... attribute_312

Class labels follow:

0 ... 199

---

## 3. Baseline Model

Image
-> pretrained CLIP ViT-B/32
-> 512-dimensional image feature
-> Linear Concept Prediction Layer
-> 312 concept scores

CLIP is used as a frozen feature extractor.

The Concept Prediction Layer is:

Linear(512, 312)

CLIP features are L2-normalized before training the concept layer.

---

## 4. Training

Training images are used to learn the Concept Prediction Layer.

Validation images are used for model selection and threshold selection.

The test set was not used for hyperparameter tuning.

The final threshold selected from the validation set is approximately:

0.36

---

## 5. Hyperparameter Tuning

Basic hyperparameter tuning was performed on the validation set.

Compared learning rates:

- 0.0001
- 0.0005
- 0.001

Compared loss weighting:

- no positive weighting
- sqrt class-imbalance weighting

A longer training experiment with a learning-rate scheduler was also tested.

The results remained stable around a validation Micro-F1 of approximately 0.53.

This indicates that further simple changes to learning rate or training duration did not substantially improve the linear baseline.

The tuning records are stored in the tuning/ folder.

---

## 6. Final Results

Validation baseline:

- Micro-F1: approximately 0.5309
- Macro-F1 with supported concepts: approximately 0.2615
- Best threshold: approximately 0.36

Final Test results:

- Precision: 0.4473
- Recall: 0.6470
- Micro-F1: 0.5289
- Macro-F1: 0.2582

Validation and Test Micro-F1 are similar, indicating that there is no large performance drop between validation and test data.

---

## 7. Prediction Files

The predictions/ folder contains:

- concept_predictions_train.pt
- concept_predictions_val.pt
- concept_predictions_test.pt

Each file contains:

- image_ids
- class_labels
- concept_targets
- concept_logits
- concept_probs
- concept_predictions
- threshold

For downstream classification, Member 4 should normally use:

concept_probs

as the input feature vector.

Each image therefore has a 312-dimensional continuous concept-score vector.

---

## 8. Model File

model/concept_layer_best.pt

This is the trained baseline Concept Prediction Layer.

Architecture:

512 -> 312 Linear Layer

---

## 9. Code

code/clip_features_trainval.py

Extracts frozen CLIP ViT-B/32 features for Train and Validation.

code/clip_features_test.py

Extracts frozen CLIP ViT-B/32 features for Test after model selection is complete.

code/train_concepts.py

Trains the baseline Concept Prediction Layer.

code/generate_concept_scores.py

Generates final Train / Validation / Test concept scores and evaluation metrics.

code/tune_cbl.py

Records the basic hyperparameter tuning experiments.

code/train_cbl_scheduler.py

Records the extended training and learning-rate scheduler experiment.

---

## 10. Interface for Member 4

Recommended downstream input:

concept_probs

Shape:

Train: [5394, 312]

Validation: [600, 312]

Test: [5794, 312]

Corresponding bird labels are stored in:

class_labels

Image identifiers are stored in:

image_ids

Member 4 should preserve the same Train / Validation / Test split and image IDs.

---

## 11. Baseline Interpretation

This implementation should be treated as the Member 2 baseline concept predictor:

Frozen global CLIP ViT-B/32 features
+
Linear Concept Prediction Layer.

It is not the complete end-to-end CBM by itself.

The downstream concept-to-class classifier is handled by Member 4.

Future localization-guided models can be compared against this baseline while keeping the dataset split and concept definitions consistent.
