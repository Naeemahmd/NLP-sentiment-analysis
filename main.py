import pandas as pd

from utils import DataIntegrityChecker,EDA
from sentiment_analysis import SentimentAnalysis
from utils import load_data

#load dataset
df = pd.read_csv("archive/Reviews.csv")
print(df.head())
print(df.columns)

# Select relevant columns for analysis
df = df[['Text', 'Score']]

# check data for null values and etc.
dic = DataIntegrityChecker(df)
integrity_report = dic.check_data_integrity()
# check outliers in score column
unique_scores = dic.get_unique_values('Score')


# Map scores to sentiment (1-3 -> Negative, 4-5 -> Positive)
df['Sentiment'] = df['Score'].apply(lambda x: 'Positive' if x > 3 else 'Negative')
df = df[['Text', 'Sentiment']]
print(f"unique number values of sentiments :", df['Sentiment'].value_counts())
print(df.head())


# Exploratory data analysis
eda = EDA(dataframe=df)
eda.describe_data()
#eda.plot_distribution_of_column(column='Sentiment')
#eda.plot_review_length_distribution(column='Text')
#eda.generate_wordcloud(column='Text')

# sentimental analysis
sa = SentimentAnalysis(dataframe=df)
sa.preprocess_data()
x_train, x_test, y_train, y_test = sa.split_data()
x_train_tfidf, x_test_tfidf = sa.Vectorize_data(x_train, x_test)
sa.build_model(x_test_tfidf, y_train)

# prediction 
y_pred = sa.predict_y(x_test_tfidf)
sa.evaluate_model(y_test=y_test,y_pred=y_pred)
sa.visualize_results(y_test,y_pred)

print('Done')