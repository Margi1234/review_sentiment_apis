from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

import pandas as pd
import numpy as np
import mysql.connector
import pickle
import os
import matplotlib.pyplot as plt

from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset
from wordcloud import WordCloud, STOPWORDS

import neuralcoref
import en_core_web_lg
import spacy

#Enter the values for you database connection
dsn_database = "db_kefdirect"  
dsn_hostname = "siteproofs.net"     
dsn_port = 3306                     
dsn_uid = "kanchan"            
dsn_pwd = "server@123" 
conn = mysql.connector.connect(host=dsn_hostname,  user=dsn_uid, passwd=dsn_pwd, db=dsn_database)
clf = pickle.load(open('clf.sav', 'rb'))
mlb = pickle.load(open('mlb.sav', 'rb'))
text_clf = pickle.load(open('aspect_clf.sav', 'rb'))

def train_model():
  basepath = os.path.dirname(os.getcwd())
  df = pd.read_csv(os.path.join(basepath,'balancedReviews.csv'))
  df = df.sample(frac=1).reset_index(drop=True)
  df = df[df['Label'].notna()]
  df = df[df.Label != 'A']
  temp_df = df.copy()
  df['Label'] = df['Label'].replace(to_replace ="N", value = -1) 
  df['Label'] = df['Label'].replace(to_replace ="P", value = 1) 
  text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),      
                     ('clf', MultinomialNB())])
  tuned_parameters = {
      'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
      'tfidf__use_idf': (True, False),
      'tfidf__norm': ('l1', 'l2'),
      'clf__alpha': [1, 1e-1, 1e-2]
  }
  x_train, x_test, y_train, y_test = train_test_split(df['Description'], df['Label'], test_size=0.33, random_state=42)
  clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
  clf.fit(x_train, y_train)
  pickle.dump(clf, open('clf.sav', 'wb'))
  return "done"

def get_sentiment(row):
  text = row['Description']
  result = clf.predict([text])
  result = result[0]
  return result
    
def overall_classification():
  train_model()
  basepath = os.path.dirname(os.getcwd())
  result = pd.read_csv(os.path.join(basepath,'reviewsLabelled.csv'))
  sql = "SELECT entity_id as ProductID, value as Product FROM catalog_product_entity_varchar WHERE attribute_id = 73"
  p_name = pd.read_sql(sql, conn)
  result = pd.merge(result, p_name, on="ProductID", how='left')
  result = result[['ReviewID', 'ProductID', 'Product', 'Title', 'Description']]
  
  result['Sentiment'] = result.apply(lambda row: get_sentiment(row), axis=1)
  result = result.sort_values(by=['ProductID'], ascending = True)
  return result

def categoryDF():
  sql = "SELECT ccp.product_id AS ProductID, ccev.value AS Category FROM catalog_category_entity_varchar as ccev LEFT JOIN catalog_category_product as ccp ON ccev.entity_id = ccp.category_id WHERE ccev.attribute_id = 45 and ccp.product_id != 'NULL' AND store_id=0"
  product_category = pd.read_sql(sql, conn)
  product_category['ProductID'] = product_category['ProductID'].astype(int)
  product_category = product_category.sort_values(by=['ProductID'], ascending=True).reset_index(drop=True)
  return product_category


@api_view(['GET'])
def checkAPI(request):
  return Response("hello")

@api_view(['GET'])
def aspectBasedMining(request):
  basepath = os.path.dirname(os.getcwd())
  annotated_reviews_df = pd.read_csv(os.path.join(basepath, 'aspectLabelled.csv'))
  annotated_reviews_df = annotated_reviews_df.sample(frac=1).reset_index(drop=True)
  annotated_reviews_df = annotated_reviews_df[annotated_reviews_df['Label'].notna()]
  def aspect_list(row):
    temp = str(row['Aspect'])
    temp2 = list(temp.split(",")) 
    return temp2
  annotated_reviews_df['Aspect'] = annotated_reviews_df.apply(lambda x: aspect_list(x), axis=1)
  nlp = spacy.load('en_core_web_lg')
  neuralcoref.add_to_pipe(nlp)
  def replace_pronouns(text):
    text = text['Description']
    doc = nlp(text)
    resolved_text = doc._.coref_resolved
    return resolved_text
  annotated_reviews_df["text_pro"] = annotated_reviews_df.apply(lambda x: replace_pronouns(x), axis=1)
  # Convert the multi-labels into arrays
  mlb = MultiLabelBinarizer()
  y = mlb.fit_transform(annotated_reviews_df.Aspect)
  X = annotated_reviews_df.text_pro

  # Split data into train and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

  # save the the fitted binarizer labels
  # This is important: it contains the how the multi-label was binarized, so you need to
  # load this in the next folder in order to undo the transformation for the correct labels.
  filename = 'mlb.sav'
  pickle.dump(mlb, open(filename, 'wb'))
  text_clf = Pipeline([('vect', CountVectorizer(stop_words = "english",ngram_range=(1, 1))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', LabelPowerset(MultinomialNB(alpha=1e-1))),])
  text_clf = text_clf.fit(X_train, y_train)
  filename = 'aspect_clf.sav'
  pickle.dump(text_clf, open(filename, 'wb'))
  data = {"data": {"code": 200, "message": "Classifier Updated"}}
  return Response(data)

@api_view(['GET'])
def aspectClassification(request):
  basepath = os.path.dirname(os.getcwd())
  test_set = pd.read_csv(os.path.join(basepath, 'balancedReviews.csv'))
  def get_aspect(row):
    temp = row['Description']
    predicted = text_clf.predict([temp])
    aspect = mlb.inverse_transform(predicted)
    return aspect[0]
  test_set['Aspect'] = test_set.apply(lambda row: get_aspect(row), axis=1)
  test_set['Sentiment'] = test_set.apply(lambda row: get_sentiment(row), axis=1)
  test_set = test_set[['ProductID', 'Aspect', 'Sentiment']]
  test_set = test_set.groupby(['Aspect', 'ProductID', 'Sentiment']).size().reset_index(name='Count')
  # test_set = test_set.groupby(['Aspect', 'ProductID']).size().reset_index(name='Count')
  # table = pd.pivot_table(test_set, values=['Count'], index=['ProductID', 'Aspect'], aggfunc=np.sum, fill_value=0)
  # table_index = table.index
  # data_dict = {}
  # for i, j in table_index:
  return Response(test_set)


@api_view(['GET'])
def getWordCloud(request):
  stopwords = set(STOPWORDS)
  result = overall_classification()
  pos_word_cloud = result.copy()
  neg_word_cloud = result.copy()
  pos_wordcloud = WordCloud(
        collocations=False,
        width=1600,
        height=800,
        background_color="white",
        stopwords=stopwords,
        max_words=100,
        random_state=40,
    ).generate(
        " ".join(
            pos_word_cloud[pos_word_cloud['Sentiment'] == 1]["Description"]
        )
    )
  neg_wordcloud = WordCloud(
      collocations=False,
      width=1600,
      height=800,
      background_color="white",
      stopwords=stopwords,
      max_words=100,
      random_state=40,
  ).generate(
      " ".join(
          neg_word_cloud[neg_word_cloud['Sentiment'] == -1]["Description"]
      )
  )
  pos_word_freq = pos_wordcloud.words_
  pos_words_list = []
  for i in pos_word_freq:
    temp = {}
    temp["tag"] = i
    temp["weight"] = int(pos_word_freq[i] * 50)
    pos_words_list.append(temp)
  neg_word_freq = neg_wordcloud.words_
  neg_words_list = []
  for i in neg_word_freq:
    temp = {}
    temp["tag"] = i
    temp["weight"] = int(neg_word_freq[i] * 50)
    neg_words_list.append(temp)
  data = {"positive_words_list": pos_words_list, "negative_words_list": neg_words_list}
  return Response(data)


@api_view(["GET"])
def createClassifier(request):
  x = train_model()
  data = {"data": {"code": 200, "message": "Classifier Updated"}}
  return Response(data)

@api_view(["GET"])
def productSentiment(request):
  result = overall_classification()
  product_sentiment_count = result.copy()
  product_sentiment_count = product_sentiment_count.groupby(['ProductID', 'Product', 'Sentiment']).size().reset_index(name="Count")
  product_sentiment_count_table = pd.pivot_table(product_sentiment_count, values='Count', index=['ProductID', 'Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = product_sentiment_count_table.index
  for j, k in p_list:
    products.append(k)
  data = {"data": {"products":products, "sentiment":product_sentiment_count_table}}
  return Response(data)

@api_view(["GET"])
def overallSentiment(request):
  result = overall_classification()
  total_sentiments = result.copy()
  total_sentiments = total_sentiments['Sentiment'].value_counts()
  lst = total_sentiments.tolist()
  positive = round((lst[0])/(lst[0] + lst[1])*100, 2) 
  negative = round((100 - positive), 2)
  data = {"data":{"positive_percent": positive, "negative_percent": negative}}
  return Response(data)

@api_view(["GET"])
def mostLikedProducts(request):
  result = overall_classification()
  product_sentiment_count = result.copy()
  product_sentiment_count = product_sentiment_count.groupby(['ProductID', 'Product', 'Sentiment']).size().reset_index(name="Count")
  liked_products = product_sentiment_count.copy()
  liked_products = liked_products[liked_products['Sentiment'] == 1].sort_values(by=['Count'], ascending=False)
  products = liked_products['Product'].tolist()
  frequency = liked_products['Count'].tolist()
  data = {"data":{"products": products, "frequency": frequency}}
  return Response(data)

@api_view(['POST'])
def getYearlyPerformance(request):
  data = request.data
  year = data['year']
  sql = "SELECT review_id as ReviewID, YEAR(created_at) as ReviewYear FROM review ORDER BY created_at"
  review_date = pd.read_sql(sql, conn)
  result = overall_classification()
  temp = result.copy()
  temp = pd.merge(temp, review_date, on='ReviewID', how='left')
  temp = temp[['ProductID', 'Product', 'Sentiment', 'ReviewYear']]
  temp = temp.groupby(['ProductID', 'Product', 'ReviewYear', 'Sentiment']).size().reset_index(name="Count")
  table = pd.pivot_table(temp, values='Count', index=['ReviewYear', 'Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  p_list = table.index
  pos = []
  neg = []
  t = {}
  products = []
  for j, k in p_list:
    if(j == year):
      products.append(k)
      t.update(table.loc[j])
  data = {"data":{"products": products, "sentiment": t}}
  return Response(data)

@api_view(["GET"])
def reviewsPerProduct(request):
  reviews_per_product = overall_classification()
  reviews_per_product = reviews_per_product.groupby(['Product']).size().reset_index(name="Count").sort_values(by=['Count'], ascending=False)
  products = reviews_per_product['Product'].tolist()
  frequency = reviews_per_product['Count'].tolist()
  data = {"data":{"products": products, "frequency": frequency}}
  return Response(data)

@api_view(["GET"])
def speakerCategory(request):
  speakers = categoryDF()
  speakers = speakers[speakers['Category'] == 'Speakers']
  result = overall_classification()
  speakers = pd.merge(speakers, result, on="ProductID", how="left")
  speakers = speakers.dropna()
  speakers = speakers.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  speakers_table = pd.pivot_table(speakers, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = speakers_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": speakers_table}}
  return Response(data)

@api_view(["GET"])
def headphonesCategory(request):
  headphones = categoryDF()
  headphones = headphones[headphones['Category'].str.contains("Headphones")]
  result = overall_classification()
  headphones = pd.merge(headphones, result, on="ProductID", how="left")
  headphones = headphones.dropna()
  headphones = headphones.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  headphones_table = pd.pivot_table(headphones, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = headphones_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": headphones_table}}
  return Response(data)

@api_view(["GET"])
def subwoofersCategory(request):
  subwoofers = categoryDF()
  subwoofers = subwoofers[subwoofers['Category'].str.contains("Subwoofers")]
  result = overall_classification()
  subwoofers = pd.merge(subwoofers, result, on="ProductID", how="left")
  subwoofers = subwoofers.dropna()
  subwoofers = subwoofers.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  subwoofers_table = pd.pivot_table(subwoofers, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = subwoofers_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": subwoofers_table}}
  return Response(data)

@api_view(["GET"])
def homeTheatreCategory(request):
  home_theatre = categoryDF()
  home_theatre = home_theatre[home_theatre['Category'].str.contains("Home Theater")]
  result = overall_classification()
  home_theatre = pd.merge(home_theatre, result, on="ProductID", how="left")
  home_theatre = home_theatre.dropna()
  home_theatre = home_theatre.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  home_theatre_table = pd.pivot_table(home_theatre, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = home_theatre_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": home_theatre_table}}
  return Response(data)

@api_view(["GET"])
def loudspeakersCategory(request):
  loudspeakers = categoryDF()
  loudspeakers = loudspeakers[loudspeakers['Category'].str.contains("Loudspeakers")]
  result = overall_classification()
  loudspeakers = pd.merge(loudspeakers, result, on="ProductID", how="left")
  loudspeakers = loudspeakers.dropna()
  loudspeakers = loudspeakers.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  loudspeakers_table = pd.pivot_table(loudspeakers, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = loudspeakers_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": loudspeakers_table}}
  return Response(data)

@api_view(["GET"])
def hifiSpeakersCategory(request):
  hifi_speakers = categoryDF()
  hifi_speakers = hifi_speakers[hifi_speakers['Category'].str.contains("Hi-Fi Speakers")]
  result = overall_classification()
  hifi_speakers = pd.merge(hifi_speakers, result, on="ProductID", how="left")
  hifi_speakers = hifi_speakers.dropna()
  hifi_speakers = hifi_speakers.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  hifi_speakers_table = pd.pivot_table(hifi_speakers, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = hifi_speakers_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": hifi_speakers_table}}
  return Response(data)

@api_view(["GET"])
def kefSpecialCategory(request):
  kef_specials = categoryDF()
  kef_specials = kef_specials[kef_specials['Category'].str.contains("KEF Specials")]
  result = overall_classification()
  kef_specials = pd.merge(kef_specials, result, on="ProductID", how="left")
  kef_specials = kef_specials.dropna()
  kef_specials = kef_specials.groupby(['Product',  'Sentiment']).size().reset_index(name='Count')
  kef_specials_table = pd.pivot_table(kef_specials, values='Count', index=['Product'], columns=['Sentiment'], aggfunc=np.sum, fill_value=0)
  products = []
  p_list = kef_specials_table.index
  for j in p_list:
    products.append(j)
  data = {"data":{"products": products, "sentiment": kef_specials_table}}
  return Response(data)