import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

df_e = pd.read_csv('geglobo_user_tweets.csv') 
df_e_2 = pd.read_csv('globoesporters_user_tweets.csv')
df_e_3 = pd.read_csv('UOLEsporte_user_tweets.csv')
df_p = pd.read_csv('EstadaoPolitica_user_tweets.csv')
df_p_2 = pd.read_csv('UOLPolitica_user_tweets.csv')
df_p_3 = pd.read_csv('OGloboPolitica_user_tweets.csv')




df_e_clean = df_e.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs', 'Unnamed: 16',
       'Unnamed: 17'])
df_e_2_clean = df_e_2.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs', 'Unnamed: 16',
       'Unnamed: 17', 'Unnamed: 18'])
df_e_3_clean = df_e_3.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs', 'Unnamed: 16'])
df_p_clean = df_p.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs', 'Unnamed: 16'])
df_p_2_clean = df_p_2.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs'])
df_p_3_clean = df_p_3.drop(columns=['Tweet Id', 'Name', 'Screen Name', 'UTC', 'Created At',
       'Favorites', 'Retweets', 'Language', 'Client', 'Tweet Type', 'URLs',
       'Hashtags', 'Mentions', 'Media Type', 'Media URLs', 'Unnamed: 16'])


df_e_clean['categoria'] = 'esporte'
df_e_2_clean['categoria'] = 'esporte'
df_e_3_clean['categoria'] = 'esporte'
df_p_clean['categoria'] = 'politica'
df_p_2_clean['categoria'] = 'politica'
df_p_3_clean['categoria'] = 'politica'


products_list_e = df_e_clean.values.tolist()
products_list_e_2 = df_e_2_clean.values.tolist()
products_list_e_3 = df_e_3_clean.values.tolist()
products_list_p = df_p_clean.values.tolist()
products_list_p_2 = df_p_2_clean.values.tolist()
products_list_p_3 = df_p_3_clean.values.tolist()


data = products_list_e + products_list_e_2 + products_list_e_3 + products_list_p + products_list_p_2 + products_list_p_3 


training, test = train_test_split(data, test_size = 0.33, random_state=42)

train_x = [i for i, j in training]
train_y = [y for x, y in training]

test_x = [x for x, y in test]
test_y = [y for x, y in test]



vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)


clf_svm = svm.SVC(kernel = 'linear')

clf_svm.fit(train_x_vectors, train_y)

test_set = [input("Manchete: ")]
new_test = vectorizer.transform(test_set)

a = clf_svm.predict(new_test)
print(a)
