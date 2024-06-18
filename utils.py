import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def load_data(path):
    df = pd.read_csv(path)
    return df

class DataIntegrityChecker:
    def __init__(self, df):
        self.df = df


    def check_data_integrity(self):
        total_rows = self.df.shape[0]
        total_columns = self.df.shape[1]
        null_values = self.df.isnull().sum().sum()
        duplicates_per_column = self.get_duplicates_per_column()
        unique_coloumn = self.df.nunique()
        data_info = self.df.info()
        statistcis = self.df.describe()
        
        integrity_report = {
            'Total Rows': total_rows,
            'Total Columns': total_columns,
            'Null Values': null_values,
            'Duplicates Per Column': duplicates_per_column,
            'Unique coloumn values': unique_coloumn,
            'Data Info': data_info,
            'Description of data': statistcis
        }
        
        print("Data check Report:")
        print(f"Total Rows: {total_rows}")
        print(f"Total Columns: {total_columns}")
        print(f"Null Values: {null_values}")
        print(f"Duplicates Per Column: {duplicates_per_column}")
        if unique_coloumn is not None:
            print(f"Unique coloumn values: {unique_coloumn}")
        print(self.df.describe())
        
        return integrity_report
    
    def get_duplicates_per_column(self):
        duplicates = {}
        for column in self.df.columns:
            duplicates[column] = self.df[column].duplicated().sum()
        return duplicates
    
    def get_unique_values(self, column_name):
        if column_name in self.df.columns:
            unique_values = self.df[column_name].unique()
            unique_values_sorted = sorted([int(value) for value in unique_values])
            print(f"Unique values in column '{column_name}': {unique_values_sorted}")
            return unique_values_sorted
        else:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
            return None


class EDA:
    def __init__(self,dataframe):
        self.dataframe = dataframe

    
    def describe_data(self):
        print(self.dataframe.describe())

    def plot_distribution_of_column(self, column='Sentiment'):
        # Distribution of sentiments
        sns.countplot(x=column, data=self.dataframe)
        plt.title(f'Distribution of {column}')
        plt.show()

    def plot_review_length_distribution(self, column= 'Text'):
        """ Calculates the length of each review by splitting the text into words and counting them.
          Uses seaborn's histplot to create a histogram of review lengths."""
        # Review length distribution
        self.dataframe['Review_Length'] = self.dataframe[column].apply(lambda x: len(x.split()))
        sns.histplot(self.dataframe['Review_Length'], bins=50)
        plt.title('Distribution of Review Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.xlim(0, 1000)  # Adjust x-axis to focus on shorter lengths for better visualization
        plt.show()


    def generate_wordcloud(self, column='Text'):
        """ visual representation of text data where the size of each word indicates its frequency or importance """
        # Word cloud for all reviews
        all_texts = ' '.join(self.dataframe[column])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_texts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {column}')
        plt.show()