# Sentiment-Detection-in-Cinematic-Reviews

# Objective
This project focuses on developing and implementing a sentiment detection system for cinematic reviews. The goal is to create a tool that can automatically analyze movie reviews and classify the sentiment expressed in them as positive, negative, or neutral. This can be particularly useful for filmmakers, marketers, and movie enthusiasts to gauge public opinion and feedback on films.

![th](https://github.com/danish1341/Sentiment-Detection-in-Cinematic-Reviews/assets/167858464/a6060f5e-ba2d-4889-90ca-eeb7b58bbbfe)


# Data Set

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
Data fields:

* id — Unique ID of each review
* review — Text of the review
* label — Sentiment of the review; 1 for positive reviews and 0 for negative reviews

# Inspect and explore data

We encountered no issues with data types and found no null values. We did, however, identify and remove some duplicate entries. The relative frequencies of the classes in the dataset are close, indicating that the data is balanced.
![Capture](https://github.com/danish1341/Sentiment-Detection-in-Cinematic-Reviews/assets/167858464/ba27cd23-c032-4108-8b9c-55372502f125)

# Data cleaning

For any NLP problem it is essential the raw data is cleaned and processed in the desired format before fed into the model. Here we have converted the data into lower case and removed punctuations using string library which proved faster in than what I had seen in the reference material. Additionally the stopwords (such as “the”, “a”, “an”, “in”) were removed using the NLTK library as these words don't matter when indexing.

Another processes we applied:

* Removed HTML tags
* Removed Urls
* Removed numbers
* Removed symbols
* Removed emojis
* Handled chat words
* Stemming / Lemmatization

# Models
## ML Algorithms

We Created a pipeline named model that contains a transformer to represent text data into vectors and a predictor.
We used different algorithms and trained them with different data and transformers. Then, we evaluate them using the accuracy and F1-Score.

Transformers:
* Bag of words
* TF-IDF
  
Predators:
* Naive Bayes
* Decision Tree
* Logistic Regression
* Random Forest
* K-Nearest Neighbor
* XGBOOST

![odel Accuracies](https://github.com/danish1341/Sentiment-Detection-in-Cinematic-Reviews/assets/167858464/a8232319-3271-4012-9f23-1bd5d479fba7


# LSTM MODEL

First, we processing the data:

* Tokenization: We used the Tokenizer class from Keras, which also takes care of converting the words to lower case and removing punctuation.
* Sequence conversion: Next, we converted the tokens into sequences of integers. Each unique word was assigned a unique integer.
* Sequence padding: Because LSTM networks require input data with the same length, we used padding to ensure all sequences had the same length.
After preprocessing our data, we moved on to constructing the LSTM network. The model was created using Keras’ Sequential API, indicating that our model would be built layer by layer in a sequential manner.

The models basic structure:

1. Embedding layer: This layer transformed our integer sequences (representing words) into dense vectors of fixed size.
2. LSTM layers: Next, we added an LSTM layer. LSTM layers are effective in processing sequences (like text), as they can capture the temporal dependencies between elements in the sequence.
3. Dense layers: After the LSTM layers, we included a dense (fully connected) layer. We used the ‘sigmoid’ activation function. We also added dropout to these layers to avoid overfitting.
4. We then trained our model on our preprocessed reviews and sentiments, using a batch size of 264 and running for 10 epochs. We also included early stopping in our training to prevent overfitting.

   ![g](https://github.com/danish1341/Sentiment-Detection-in-Cinematic-Reviews/assets/167858464/ac7a4637-f25f-4615-a455-bd2d80bcd631)

# Further Work

Using an LSTM-based neural network, we achieved impressive results in predicting our target. Future work might involve exploring different neural network architectures, using pre-trained embeddings, or applying this model to other NLP tasks.




