# NLP-sentiment-analysis-customer-review
Analyzing customer sentiment from product reviews to improve product features and customer service

You can clone this repo from: https://github.com/Naeemahmd/NLP-sentiment-analysis.git

# Project Structure
- Data Collection
- Data Preprocessing
- Feature Extraction
- Model Building
- Evaluation
- Visualization

The dataset used is Amazon fine food reviews: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.

we are doing a binary classifictaion problem by taking reviews 1,2,3 as negative and 4,5 as positive

# virtual environment setup
To build and install the module in a virtual environment, execute the following commands in the project's root directory

```ruby
# create a virtual enviroment     
python3 -m venv .venv

# active the environment
source .venv/bin/activate       # for Linux, Mac
.venv/Scripts/activate          # for Windows

#install required packages
pip install -r requirements.txt
# to update required packages
pip freeze > requirements.txt

```
