import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import streamlit as st
import umap

#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
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
Upload your own CSV file below or continue using the example Pima Diabetes dataset.
''')
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
else:
    st.info('Awaiting for CSV file to be uploaded. Now using the example Pima Diabetes dataset below.')
    @st.cache
    def load_data():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        pima = pd.read_csv(url, names=names)
        return pima
        #titanic = sns.load_dataset('titanic')
        #titanic = titanic.copy()
        #return titanic
    df = load_data()
st.write('---')

# Explore data
st.header('**Explore data**')
st.write('Below you can see the first 10 rows of the dataset.')
st.write(df.astype('object').head(10))
#st.write(df.dtypes)
st.write('Descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding missing values:')
st.write(df.describe(include = 'all'))

st.write('Count total missing in each column:')
missing = df.isnull().sum().to_frame(name = 'missing').T
percent_missing = df.isnull().sum() * 100 / len(df)
percent_missing = percent_missing.to_frame(name = '%missing').T
st.write(pd.concat([missing, percent_missing]))

st.write('Press the button below to generate a pair plot:')
if st.button('Generate Seaborn Pairplot'):
    fig = sns.pairplot(df, diag_kind="kde")
    fig.map_lower(sns.kdeplot, levels=4, color=".2")
    st.pyplot(fig)

#st.write('Press the button below to generate a full exploratory data analyses report (using Pandas Profiling).')
#if st.button('Generate Pandas Profiling Report'):
#    pr = ProfileReport(df.astype('object'), explorative=True)
#    st_profile_report(pr)

st.write('---')

# Preprocess data
st.header('**Preprocess data**')
st.subheader('Select features')
st.write('Select features / variables that will go into anomaly detection model:')
features = st.multiselect('Selected features:', options = df.columns.tolist(), default = df.columns.tolist())
df_model = df[features]
df_id = df[df.columns[~df.columns.isin(features)]]
#df_id.insert(0, 'ID', range(0, len(df_id)))

st.write('**Model features** in selection, column types and first 5 rows:')
st.write(df_model.dtypes.to_frame(name = 'dtypes').T)
st.write(df_model.astype('object').head(5))

st.write('**Not** in selection (assumed ID variables), column types and first 5 rows:')
st.write(df_id.dtypes.to_frame(name = 'dtypes').T)
st.write(df_id.astype('object').head(5))
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

st.write('Missing data handling results (first 10 rows):')
st.write(df_model.astype('object').head(10))

st.write('Descriptive statistics after handling of missing data:')
st.write(df_model.describe(include = 'all'))

st.write('Count total missing in each column after missing data handling:')
missing = df_model.isnull().sum().to_frame(name = 'missing').T
percent_missing = df_model.isnull().sum() * 100 / len(df_model)
percent_missing = percent_missing.to_frame(name = '%missing').T
st.write(pd.concat([missing, percent_missing]))
st.write('---')

st.subheader('One hot encoding')
# One hot encoding for categorical variables
cats = df_model.dtypes == 'object'
le = LabelEncoder() 
for x in df_model.columns[cats]:
    sum(pd.isna(df_model[x]))
    df_model.loc[:,x] = le.fit_transform(df_model[x])
onehotencoder = OneHotEncoder() 
df_model.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(df_model.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names()))
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
st.write('The selected features are standardized using the StandardScaler:')
st.write(X.head())
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
    
    st.write('Top 10 anomaly scores for the', model, 'model:')
    df_id['scores'] = scores
    top10 = df_id.nlargest(10, 'scores')
    st.write(top10)
with col2:
    # UMAP
    @st.cache
    def make_umap(data):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data)
        df = pd.DataFrame(embedding)
        #df['anomaly'] = 'other'
        return df
    df_umap = make_umap(data = X)

    fig_umap = px.scatter(
        df_umap,
        x=0,
        y=1,
        #color="anomaly",
        title="uMAP Plot with Outliers",
        #hover_data=[df_umap.index],
        opacity=0.7
        )

    top10_list = top10.index.tolist()
    #df_umap['anomaly'][top10_list] = 'anomaly'

    fig_umap.add_scatter(
        x=df_umap[0][top10_list],
        y=df_umap[1][top10_list],
        #x=0,
        #y=1,
        color_discrete_sequence=["red"],
        #color="anomaly",
        hover_data=[df_umap.loc[top10_list].index],
        opacity=0.7
        )
    st.write(fig_umap.show())
st.write('---')


# Interpret anomalies
st.header('**Interpret individual anomalies**')
st.write('Select an individual top 10 anomaly to view the position of the anomaly in histograms.')
anomaly_select = st.selectbox('Selected index number', options = (top10_list)) 

df_ano = df.copy()
df_ano['anomaly'] = 'other'
df_ano['anomaly'][anomaly_select] = 'anomaly'

x_axes = st.selectbox('Selected variable on X axes', options = (df_ano.columns)) 
y_axes = st.selectbox('Selected variable on Y axes', options = (df_ano.columns)) 


fig_hist = px.histogram(df_ano,
    x=df_ano[x_axes],
    color='anomaly')
fig_hist.add_vline(x=df_ano[x_axes][anomaly_select], line_width=3, line_dash="dash", line_color="#f63366")
fig_scatter = px.scatter(df_ano,
    x=df_ano[x_axes],
    y=df_ano[y_axes],
    color='anomaly')

col3, col4 = st.beta_columns(2)
with col3:
    st.write(fig_hist)
with col4:
    st.write(fig_scatter)

#fig_ano = sns.pairplot(df_ano, hue='anomaly', diag_kind='hist')
#st.pyplot(fig_ano)

st.write('---')