import re
import pandas as pd
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

#pd.options.display.max_columns = None
#pd.set_option('display.max_rows', 100)

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

replace_dict = {'count of': 'number of',
                'num of': 'number of',
                'no of': 'number of',
                'no. of': 'number of',
                'variable':'column',
                'col':'column',
                'feature':'column',
                'record':'row',
                'observation':'row',
                'sample':'row',
                'get':'print',
                'display':'print',
                'show':'print',
                'distinct':'unique',
                'repeating':'duplicate',
                'repeatitive':'duplicate',
                'identical':'duplicate',
                'matching':'duplicate',
                'na':'missing',
                'nan':'missing',
                'ending':'last'}

# read intent templates
intent_df = pd.read_csv("intent.csv")

# convert intent templates to tfidf vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(intent_df['queries'])
tfidf_dense = tfidf.todense()

# get feature-wise missing data
def intent_1(df):
    st.write("Column-wise missing data...\n")
    st.write(df.isnull().sum())

# get features names
def intent_2(df):
    st.write("Column names...\n")
    st.markdown(list(df.columns))
    st.markdown(f"There are {len(df.columns)} columns")

# get non-numeric features
def intent_3(df):
    st.write("Non numeric columns...\n")
    st.markdown(list(df.select_dtypes(include='object').columns))
    st.markdown(f"There are {len(df.select_dtypes(include='object').columns)} non-numeric columns")

# get numeric features
def intent_4(df):
    st.write("Numeric columns...\n")
    st.markdown(list(df.select_dtypes(exclude='object').columns))
    st.markdown(f"There are {len(df.select_dtypes(exclude='object').columns)} numeric columns")

# get unique items in input features
def intent_5(df, cols):
    for i in cols:
        st.write(df[i].unique())

# get distribution of distinct items in input features
def intent_6(df, cols):
    for i in cols:
        st.write(df[i].value_counts())
    

# get first n features
def intent_7(df, query_text):
    col_count = int(re.findall("\d+", query_text)[-1])
    st.write("first "+str(col_count)+" columns...")
    st.write(df.iloc[:,:col_count])

# get last n features
def intent_8(df, query_text):
    col_count = int(re.findall("\d+", query_text)[-1])
    st.write("last "+str(col_count)+" columns...")
    st.write(df.iloc[:,-col_count:])

# get a range of features
def intent_9(df, query_text):
    col_indices = re.findall("\d+", query_text)
    col_indices = sorted(list(map(int, col_indices)))

    st.write("column "+str(col_indices[0])+" to column "+str(col_indices[1])+"...")
    st.write(df.iloc[:,col_indices[0]-1:col_indices[1]])

# get first n rows
def intent_10(df, query_text):
    col_count = int(re.findall("\d+", query_text)[-1])
    st.write("first "+str(col_count)+" rows...")
    st.write(df.loc[:col_count-1,:])

# get last n rows
def intent_11(df, query_text):
    col_count = int(re.findall("\d+", query_text)[-1])
    st.write("last "+str(col_count)+" rows...")
    st.write(df.iloc[-col_count:,:])

# get a range of rows
def intent_12(df, query_text):
    col_indices = re.findall("\d+", query_text)
    col_indices = sorted(list(map(int, col_indices)))

    st.write("row "+str(col_indices[0])+" to row "+str(col_indices[1])+"...")
    st.write(df.iloc[col_indices[0]-1:col_indices[1],:])

# get duplicate rows
def intent_13(df):
    st.write("Number of rows with duplicates: ",str(len(df[df.duplicated()])))
    st.write(df[df.duplicated()])

# get variable distribution
def intent_14(df, cols):
    if df[cols[0]].dtypes == 'O':
        st.write(df[cols[0]].value_counts())
    else:
        st.write(df[cols[0]].quantile([0, 0.25, 0.5, 0.75, 1]))
        st.pyplot(df[cols[0]].hist(bins = 30).plot())

# get rows based on a condition
def intent_15(df, query_text):
    ops_list = ["==","<",">","<=",">=","!="]
    query_text = query_text.split()
    q_ops = list(set(ops_list) & set(query_text))
    
    if len(q_ops) > 1:
        st.write("Found multiple conditions. Please enter single condition only.")
        
    else:
        q_ops_idx = query_text.index(q_ops[0])
        feature = query_text[q_ops_idx-1]
        value = query_text[q_ops_idx+1]
        if (len(re.findall("\d+",value))>0):
            if len(value) == len(re.findall("\d+",value)[0]):
                query_code = f"df[df[{feature!r}]{q_ops[0]}{value}]"
        else:
            query_code = f"df[df[{feature!r}]{q_ops[0]}{value!r}]"
        
        st.markdown(f"Found {len(eval(query_code))} rows")
        st.write(eval(query_code))


class Pandas_NLI:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.columns = dataframe.columns

    def get_feature_names(self, toks):
        mentions = []
        # search tokens for feature names
        for i in toks:
            if i in self.columns:
                mentions.append(i)
        
        return mentions

    def get_intent(self, input_text, feats):
        # get tfidf vector of the query
        q = tfidf_vectorizer.transform([input_text]).todense()

        # get the most similar intent template
        sorted_indices = cosine_similarity(q, tfidf_dense).argsort()
        intent_num = intent_df.loc[sorted_indices[:,-3:][0]]["intent"].mode().values[0]

        if intent_num == 1:
            intent_1(self.dataframe)

        elif intent_num == 2:
            intent_2(self.dataframe)

        elif intent_num == 3:
            intent_3(self.dataframe)

        elif intent_num == 4:
            intent_4(self.dataframe)

        elif intent_num == 5:
            intent_5(self.dataframe, feats)

        elif intent_num == 6:
            intent_6(self.dataframe, feats)

        elif intent_num == 7:
            return intent_7(self.dataframe, input_text)

        elif intent_num == 8:
            return intent_8(self.dataframe, input_text)

        elif intent_num == 9:
            return intent_9(self.dataframe, input_text)
        
        elif intent_num == 10:
            return intent_10(self.dataframe, input_text)

        elif intent_num == 11:
            return intent_11(self.dataframe, input_text)

        elif intent_num == 12:
            return intent_12(self.dataframe, input_text)

        elif intent_num == 13:
            return intent_13(self.dataframe)

        elif intent_num == 14:
            return intent_14(self.dataframe, feats)

        else:
            return intent_15(self.dataframe, input_text)

    def query(self, q):
        # take off punctuation marks from the user query
        q = q.translate(str.maketrans('', '', '"#$&\'()*+,-./:;?@[\\]^_`{|}~'))

        # replace words and phrases
        for pattern, substitute in replace_dict.items():
            q = q.replace(pattern, substitute)

        tokens = q.split()
        input_features = self.get_feature_names(tokens)
        return self.get_intent(q, input_features)
