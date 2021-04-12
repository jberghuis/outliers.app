import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

st.set_page_config(page_title="Outliers App", page_icon=None, layout="wide", initial_sidebar_state="auto")

# Web App Title
st.title('**Outliers App**')
st.markdown('''
### **Multivariate Unsupervised Outlier Detection**  
created by Jonathan Berghuis.  
  
Outlier detection broadly refers to the task of identifying observations which may be considered anomalous given the distribution of a sample.  
The aim of this app is to explore the top 10 results of different outlier / anomaly detection algorithms using the [PyOD library](https://pyod.readthedocs.io/en/latest/).  
PyOD is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. 

An **unsupervised** approach is used here, meaning unlabelled training data is assumed. The ground truth (inlier vs outlier) is considered unavailable.     
''')
st.write('---')

# Select data
st.header('**Select data**')
st.markdown('''
Upload your own CSV file below or continue using the example Titanic dataset.
''')
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
else:
    st.info('Awaiting for CSV file to be uploaded. Now using the example Titanic dataset below.')
    @st.cache
    def load_data():
        titanic = pd.read_csv('train.csv')
        return titanic
    df = load_data()
st.write('---')

# Explore data
st.header('**Explore data**')
st.write('Below you can see the first 50 rows of the dataset.')
st.write(df.astype('object').head(50))

st.write('Column types, descriptive statistics and missing values:')
coltypes = df.dtypes.to_frame(name = 'dtypes')
coldescribe = df.describe(include = 'all').round(2).T

missing = df.isnull().sum().to_frame(name = 'missing')
percent_missing = df.isnull().sum() * 100 / len(df)
percent_missing = percent_missing.to_frame(name = '%missing').round()
colmissing = pd.concat([missing, percent_missing], axis=1)

st.write(pd.concat([coltypes, coldescribe, colmissing], axis=1).round(2))
st.write('---')

# Preprocess data
st.header('**Preprocess data**')
st.subheader('Select features')
st.write('**Select features**, variables that will go into anomaly detection model.')
features = st.multiselect('Selected features:', options = df.columns.tolist(), default = df.columns.tolist())
df_model = df[features]

notfeatures = list(set(df.columns.tolist()).difference(features))
idvars = []
if notfeatures:
    st.write('**Select ID variables**:')
    idvars = st.multiselect('Selected ID variables:', options = notfeatures, default = notfeatures)
    removedvars = list(set(notfeatures).difference(idvars))
    if removedvars:
        st.write('**Removed variables**:')
        st.markdown(removedvars)
df_id = df[idvars]
st.write('---')

st.subheader('Missing values')
st.write('Select how to handle missing values. If imputation is chosen, then categoricals are imputed using the most frequent category.')
missing_option = st.selectbox('Missing values', options = ('Drop rows with missing values', 'Simple mean imputation', 'Simple median imputation'), index = 2) 

if missing_option == 'Drop rows with missing values':
    bool_idx = df_model.isnull().any(axis=1)
    df_model = df_model.dropna()
    df_id = df_id[~bool_idx]
    # check if no. of rows is the same
    st.write('Numbers of rows left after dropping rows with missing values:', df_model.shape[0], '/', df.shape[0], '(', round(df_model.shape[0]/df.shape[0]*100),'%)')
else:
    #@st.cache
    def impute_missing(data):
        # Impute nans with mean or median for numerics and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(df_model.loc[:,data.dtypes == 'object'].columns) != 0:
            data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
        if len(df_model.loc[:,data.dtypes == 'category'].columns) != 0:
            data.loc[:,data.dtypes == 'category'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'category'])
        if missing_option == 'Simple mean imputation':
            imp = SimpleImputer(missing_values = np.nan, strategy="mean")
        else:
            imp = SimpleImputer(missing_values = np.nan, strategy="median")
        data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])
        return data
    df_model = impute_missing(data = df_model)

st.write('---')

st.subheader('One hot encoding')
# One hot encoding for categorical variables
cats = df_model.dtypes == 'object'
df_model = pd.get_dummies(df_model, columns=df_model.columns[cats].tolist(), drop_first=True)
st.write('Dummies are created for the selected categorical features (One hot encoding):')
st.write(df_model.head())
st.write('---')

st.subheader('Standardization')
# Standardize the feature data
X = df_model.copy()
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = df_model.columns
st.write('The selected features are standardized using the StandardScaler.')
st.write('---')

st.subheader('**Data after preprocessing**')
# Standardize the feature data
st.write('Below you can see the first 50 rows of the dataset.')
st.write(X.head(50))

st.write('Column types, descriptive statistics and missing values:')
coltypes = X.dtypes.to_frame(name = 'dtypes')
coldescribe = X.describe(include = 'all').round(2).T

missing = X.isnull().sum().to_frame(name = 'missing')
percent_missing = X.isnull().sum() * 100 / len(X)
percent_missing = percent_missing.to_frame(name = '%missing').round()
colmissing = pd.concat([missing, percent_missing], axis=1)

st.write(pd.concat([coltypes, coldescribe, colmissing], axis=1).round(2))
st.write('---')


# Run outlier detection models
st.header('**Fit outlier detection model**')

col1, col2 = st.beta_columns(2)
with col1:
    rndm = st.number_input('Random number for reproducibility', value = 42)
    # To ensure the results are reproducible
    np.random.seed(rndm)

    model = st.selectbox('Outlier / anomaly detection algorithm:', 
        options = ('IForest', 'FeatureBagging', 'PCA', 'MCD', 'OCSVM', 'LOF', 'CBLOF', 'HBOS', 'KNN', 'ABOD', 'COPOD'), 
        index = 0)
    # train the COPOD detector
    if model == 'IForest':
        clf = IForest()
    elif model == 'FeatureBagging':
        clf = FeatureBagging()
    elif model == 'PCA':
        clf = PCA()
    elif model == 'MCD':
        clf = MCD()
    elif model == 'OCSVM':
        clf = OCSVM()
    elif model == 'LOF':
        clf = LOF()
    elif model == 'CBLOF':
        clf = CBLOF()
    elif model == 'HBOS':
        clf = HBOS()
    elif model == 'KNN':
        clf = KNN()
    elif model == 'ABOD':
        clf = ABOD()
    else:
        clf = COPOD()
    

    # fit the model
    clf.fit(X)
    # get outlier scores
    scores = clf.decision_scores_  # raw outlier scores

with col2:
    st.write('Top 10 anomaly scores for the', model, 'model:')
    df_id.loc[:,'scores'] = scores
    top10 = df_id.nlargest(10, 'scores')
    top10_list = top10.index.tolist()
    st.write(top10)

st.write('---')

# Interpret anomalies
st.header('**Interpret individual anomalies**')
if st.button('Interpret individual anomalies'):
    st.write('Select an individual top 10 anomaly to view the position of the anomaly in histograms.')
    
    col3, col4 = st.beta_columns(2)
    with col3:
        anomaly_select = st.selectbox('Selected index number', options = (top10_list)) 
        df_ano = df.copy()
        df_ano['anomaly'] = 'other'
        df_ano.loc[anomaly_select,'anomaly'] = 'anomaly'

        x_axes = st.selectbox('Selected variable on X axes', options = (features)) 
        y_axes = st.selectbox('Selected variable on Y axes', options = list(set(features).difference(x_axes))) 
    with col4:
        # UMAP
        @st.cache
        def make_umap(data):
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(data)
            df = pd.DataFrame(embedding)
            return df
        df_umap = make_umap(data = X)

        fig_umap = px.scatter(
            df_umap,
            x=0,
            y=1,
            title="uMAP Plot with Outlier",
            opacity=0.7
            )

        st.write(df_umap.loc[anomaly_select,])

        fig_umap.add_trace(
            go.Scatter(
                x=anomaly_select[0],
                y=anomaly_select[1],
                #x=df_umap.loc[[anomaly_select], 0],
                #y=df_umap.loc[anomaly_select, 1],
                mode='markers',
                opacity=1
                )
            )
        st.write(fig_umap)

    col5, col6 = st.beta_columns(2)
    with col5:
        fig_hist = px.histogram(df_ano,
        x=df_ano[x_axes],
        color='anomaly')
        fig_hist.add_vline(x=df_ano[x_axes][anomaly_select], line_width=3, line_dash="dash", line_color="#f63366")
        st.write(fig_hist)
    with col6:
        fig_scatter = px.scatter(df_ano,
        x=df_ano[x_axes],
        y=df_ano[y_axes],
        color='anomaly')
        st.write(fig_scatter)

    st.write('---')