# Arabic Punctuation Restoration

An individual NLP project developed for the Daal Challenge to restore punctuation in Arabic text while preserving the original word sequence.

## Approach

- Fine-tune `aubmindlab/bert-base-arabertv02` for multi-label token classification.
- Predict seven punctuation marks: `.`, `،`, `؟`, `!`, `:`, `؛`, and `-`.
- Align word targets to the last subtoken and split long texts into overlapping windows.
- Train one model using weighted binary cross-entropy and another using focal loss.
- Combine both models with class-specific prediction thresholds.
- Generate pseudo-labels for the unlabeled test inputs, select approximately the highest-confidence 20%, and fine-tune the BCE model further.
- Check that final predictions preserve the input word sequence and export a submission CSV.

## Technologies

Python, PyTorch, Hugging Face Transformers and Datasets, AraBERT, Pandas, and Scikit-learn.

## Files

- `arabic_punctuation_restoration.ipynb`: training, inference, and submission workflow.
- `requirements.txt`: required Python packages.

Competition datasets are not included due to sharing restrictions. Running the notebook requires authorized access to the training and test data. No dataset samples, trained model weights, or prediction files are included.

## Getting Started

1. Clone the repository or download and extract its ZIP archive.
2. Install dependencies with `pip install -r requirements.txt`.
3. Open the notebook in Jupyter or Google Colab.
4. If you have authorized access to the competition data, place `train.csv` and `test.csv` privately in the notebook working directory. The expected columns are `id`, `text`, and `final_text` for training, and `id` and `text` for testing. Do not commit these files. A GPU is recommended for training.
5. Run the cells in order. Initial model loading requires an internet connection.

The cleaned workflow trains and saves both models under `models/`; pretrained competition checkpoints are not included. It generates `submission_overlap40.csv` containing `id` and `final_text`. The filename is retained from the original experiment; the final prediction function currently defaults to 20 overlapping words.

## Evaluation and Reproducibility

The supplied competition metric calculates macro-F1 across the seven punctuation classes. This experiment uses all training rows and does not create a held-out validation set. Test gold labels are unavailable.

The original notebook filename was `0_674.ipynb`, but no saved outputs or leaderboard evidence were provided to verify that score. No verified score is reported here.

Pseudo-labeling uses the unlabeled competition test inputs. This is a transductive experiment, rather than an evaluation on an untouched test set.

This version corrects outdated explanatory text, saves the BCE checkpoint before loading it, replaces personal Drive paths with local paths, and moves the alignment check after prediction generation. Notebook format and Python syntax were checked; full GPU training and score reproduction were not run.

## Author

[Khadijah Alshehri](https://github.com/ka79023)
