# Construction-and-Evaluation-of-a-Multi-task-Text-Retrieval-and-Ranking-Model

This repository contains the code and experiments for constructing and evaluating a multi-task model that performs both text retrieval and ranking. The project leverages multi-task learning to jointly optimize for retrieving relevant documents and ranking them based on relevance, providing a unified approach to information retrieval challenges.

## Dataset and Data Format

- **Training Data:** `train_data.tsv`  
  This file is given, contains records in the format:  
  `<qid> <pid> <query> <passage> <relevancy>`

- **Validation Data:** `validation_data.tsv`  
  This file is given, used for model evaluation with ground truth relevance scores.

- **Test Data:**  
  - `test-queries.tsv`: Contains queries for final testing.  
  - `candidate_passages_top1000.tsv`: Lists up to 1000 candidate passages per query.

## Evaluation Metrics

The models are evaluated based on two key metrics:
- **Mean Average Precision (mAP):**  
  Measures how well the system ranks relevant passages across queries.
- **Normalized Discounted Cumulative Gain (nDCG):**  
  Considers both the relevance and the ranking position of retrieved passages.

## Tasks and Methods

### Task 1: BM25 Baseline
- **Objective:** Establish a baseline using the BM25 retrieval model.
- **Approach:**
  - **Preprocessing:** Convert text to lowercase, remove punctuation, and perform tokenization and lemmatization.
  - **Inverted Index:** Construct an inverted index mapping tokens to passages.
  - **Scoring:** Compute BM25 scores for ranking passages.
  - **Evaluation:** Assess ranking performance using mAP and nDCG at different cutoffs (e.g., Top-20 and Top-100).

### Task 2: Logistic Regression (LR)
- **Objective:** Implement a logistic regression model for re-ranking passages.
- **Approach:**
  - **Data Subsampling:** Retain all positive examples and sample 5% of negative examples.
  - **Embedding Preparation:** Train a Word2Vec model on a unified corpus of queries and passages, generating 100-dimensional embeddings.
  - **Feature Construction:** Concatenate the averaged query and passage embeddings to form a 200-dimensional feature vector.
  - **Training:** Utilize gradient descent with binary cross-entropy loss.
  - **Evaluation:** Rank passages and compute mAP and nDCG.

### Task 3: LambdaMART (LM)
- **Objective:** Improve ranking performance using a learning-to-rank approach.
- **Approach:**
  - **Feature Engineering:** Create a 3-dimensional feature vector for each query–passage pair consisting of:
    - The dot product of query and passage embeddings.
    - The Euclidean norm of the query embedding.
    - The Euclidean norm of the passage embedding.
  - **Model Training:** Use XGBoost with the `rank:ndcg` objective and perform 5-fold cross-validation for hyperparameter tuning.
  - **Evaluation:** Measure the final model performance using mAP and nDCG.

### Task 4: Neural Network (NN)
- **Objective:** Leverage a neural network to capture non-linear interactions for re-ranking.
- **Approach:**
  - **Architecture:** Design a feed-forward neural network with:
    - Three hidden layers (256, 128, 64 units) and a dropout rate of 0.5.
  - **Training:**  
    - Utilize the Adam optimizer with a learning rate of 0.0001 and weight decay of 4e-4.
    - Train for 100 epochs using mini-batch gradient descent (batch size: 512).
    - Apply BCEWithLogitsLoss for stable convergence.
  - **Evaluation:** Generate relevance scores, rank passages, and calculate mAP and nDCG.

## Final Submission

The project produces three output files for the test queries, each corresponding to one of the re-ranking models:
- **`LR.txt`** – Results from the Logistic Regression model.
- **`LM.txt`** – Results from the LambdaMART model.
- **`NN.txt`** – Results from the Neural Network model.

Each file contains up to 100 ranked passages per query, adhering to the required format.

## Conclusion

This project provides a comprehensive exploration of passage retrieval using a spectrum of techniques:
- A **BM25 baseline** for initial relevance scoring.
- **Logistic Regression** for leveraging averaged word embeddings.
- **LambdaMART** for advanced ranking with a compact 3D feature set.
- A **Neural Network** to model complex non-linear interactions.

For detailed methodology, experimental setups, and analysis, please refer to the full project report (`IRDM.pdf`) and the implementation in the Jupyter Notebook (`coursework2.ipynb`).
