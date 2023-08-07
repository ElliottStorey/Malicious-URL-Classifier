# Malicious URL Classifier
Malicious URL classifier implemented using Pytorch with a streamlit frontend. Trained on [Malicious_n_Non-Malicious URL](https://www.kaggle.com/datasets/antonyj453/urldataset), [Malicious URLs dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset), and [Malicious And Benign URLs](https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls).

## Usage
#### Initialize and train Word2Vec model
```
python3 word2vec.py
```
#### Train classifier (optional)
```
python3 train.py
```
#### Run streamlit app
```
streamlit run streamlit_app.py
```