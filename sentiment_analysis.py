import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalysis:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.vectorizer = None
        self.model = None

    def preprocess_text(self,text):
        # remove all non word character with a space, \w matches any character that is not a word character
        text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
        # remove single characters surrounded by space, \s+ matches one or more whitespace characters, 
        # [a-zA-Z] matches any single letter
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters surrounded by spaces
        # remove single characters from starting of a string, \^ matches the start of the string
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from the start
        # Substitute multiple spaces with a single space, ensuring uniform spacing in the text. flags=re.I makes the regular expression case-insensitive,
        text = re.sub(r'\s+', ' ', text, flags=re.I)  # Substitute multiple spaces with a single space
        text = text.lower()  # Convert to lowercase

        #  Tokenize the text, splits the text into individual words
        tokens = nltk.word_tokenize(text)  # Tokenize the text
        # Remove stopwords, removes common stopwords (like 'the', 'and', 'is', etc.) from the token list
        tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]  # Remove stopwords

        # Join tokens back into a single string
        return ' '.join(tokens)
        
    def preprocess_data(self):
        """ clean each row of text """
        self.dataframe['Cleaned_Text'] = self.dataframe['Text'].apply(self.preprocess_text)
        print("Data Preprocessing Completed")
        
        # Check if any preprocessed text is empty
        empty_texts = self.dataframe[self.dataframe['Cleaned_Text'].str.strip() == '']
        if not empty_texts.empty:
            print("Warning: Found empty texts after preprocessing:")
            print(empty_texts.head(10))  # Print the first 10 empty preprocessed texts for inspection
        
        # Remove rows with empty preprocessed text
        self.dataframe = self.dataframe[self.dataframe['Cleaned_Text'].str.strip() != '']
        print("Removed empty rows after preprocessing")
        print(f"Remaining rows: {len(self.dataframe)}")

        return self.dataframe

    def split_data(self):
        x = self.dataframe['Cleaned_Text']
        y = self.dataframe['Sentiment']

        # Perform the train-test split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Debugging: Print the shapes of the split data
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    
    def Vectorize_data(self, x_train, x_test):
        """  converting text data into numerical features that capture the importance 
        of words in a document relative to a corpus. limits the number of features (words)
        to the top 5000 most important ones, reducing dimensionality and computational cost. """
        self.vectorizer = TfidfVectorizer(max_features=5000)
        # Debugging: Print out a few samples of the training data
        print("Sample training data before vectorization:")
        print(x_train.head(5))
        
        X_train_tfidf = self.vectorizer.fit_transform(x_train)
        X_test_tfidf = self.vectorizer.transform(x_test)
        
        # Debugging: Print the shapes of the resulting matrices
        print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
        print(f"X_test_tfidf shape: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def build_model(self, x_train, y_train):
        """ Logistic Regression is a widely used algorithm for binary classification problems """
        self.model = LogisticRegression()

         # Debugging: Ensure that the training data and labels are aligned
        print(f"X_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print("Started training of model")
        self.model.fit(x_train,y_train)
        print("Model training completed")

    def predict_y(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def evaluate_model(self, y_test, y_pred):
        print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=y_pred))
        print("Classififcation report: ", )
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        return y_pred
    

    def visualize_results(self, y_test, y_pred):
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.title('Distribution of Review Lengths')
        plt.xlabel('Number of Words')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        positive_text = ' '.join(self.dataframe[self.dataframe['Sentiment'] == 'Positive']['Cleaned_Text'])
        wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_positive, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Positive Reviews')
        plt.show()

        negative_text = ' '.join(self.dataframe[self.dataframe['Sentiment'] == 'Negative']['Cleaned_Text'])
        wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_negative, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Negative Reviews')
        plt.show()
