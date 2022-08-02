"""
@author: jishnu
"""
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import streamlit.components.v1 as components


st.set_page_config(page_title='Analysis on Research Papers',  layout='wide', page_icon=':computer:')
st.set_option('deprecation.showPyplotGlobalUse', False)

base_df = pd.read_csv("tsne_scopus_arxiv_small12.csv", low_memory=False)
### Functions ###
def gen_wordcloud(text):
    wordcloud = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(text)
    #plot the wordcloud
    plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud)
    #to remove the axis value
    plt.tight_layout(pad=0)
    plt.axis("off")
    plt.show()

def eda_from_feature(feature, comb_df, threshold=150):
    all_f = ",".join(comb_df[feature].dropna().astype(str)).split(",")
    uniq_f = list(set(all_f))
    f_df = pd.DataFrame()
    f_df[feature] = uniq_f
    f_df['counts'] = f_df[feature].apply(lambda x: all_f.count(x))
    f_df['perc'] = f_df.counts.apply(lambda x: str(round(x/len(comb_df)*100,2))+'%')
    f_df.sort_values(by='counts', ascending=False, inplace=True)
    f_df = f_df.loc[f_df['counts']>threshold]
    f_df.reset_index(drop=True)
    return f_df, all_f

def make_word_cloud(sub, df, contain=True, freq=False):
    """
    Expects clean text to generate wordcloud
    """
    if contain:
        df = df.loc[df['subject'].str.contains(sub)]
    else:
        df = df.loc[df['subject']==sub]
    text = " ".join(df.clean)
    if len(text)<=3:
        print(F"No text for {sub}")
        return 
    words = word_tokenize(text)
    if freq:
        fdist = FreqDist(words)
        fdist.plot(20)
        plt.show()
    #generating the wordcloud
    gen_wordcloud(text)

def breakDown(sub, uniqSub, df):
    out = {}
    dat = df.loc[df.subject.str.contains(sub)]
    u_Sub = [i for i in uniqSub if i!=sub]
    out[sub] = len(dat.loc[dat['subject']==sub])
    other = ''
    other_count = 0
    for i in u_Sub:
        val = len(dat.loc[dat.subject.str.contains(i)])
        if val >= 0.02*len(dat):
            out[i] = val
        else:
            other += i + ','
            other_count += val
    out[other] = other_count
    return out

#this is the header
st.markdown("<h1 style='text-align: center; color: grey;'>Topic Modelling and Visualisation</h1>", unsafe_allow_html=True)
st.text('')

data = st.sidebar.multiselect('Choose API', ['Scopus', 'Arxiv'], default=['Scopus', 'Arxiv'], help = 'Filter report to show results based on API')
scopus = 'Scopus' in data
arxiv = 'Arxiv' in data

if scopus and not arxiv:
    eda_df = base_df.loc[base_df['api_flag']==0].copy()
elif arxiv and not scopus:
    eda_df = base_df.loc[base_df['api_flag']==1].copy()
else:
    eda_df = base_df

with st.spinner('Updating Report...'):
    eda_df['affiliation_country'].fillna('Missing Info', inplace=True)
    eda_df['type'].fillna('Missing Info', inplace=True)
    eda_df['scopus_info'].fillna('Missing Info', inplace=True)
    eda_df['cited_by'].fillna(0, inplace=True)

    countries, all_c = eda_from_feature('affiliation_country', eda_df, 300)
    subjects, allSub = eda_from_feature('subject', eda_df)

    country_df = countries.loc[countries['affiliation_country']!='Missing Info']

    subj = st.sidebar.multiselect('Choose Subject', subjects.subject.values.tolist(), help = 'Filter report to show info on subject level')
    year = st.sidebar.multiselect('Choose Year', eda_df.year.unique().tolist(), default=[], help = 'Filter report with year')
    country = st.sidebar.multiselect('Choose Country', countries.affiliation_country.values.tolist(), default=[], help = 'Filter report to show info on country level')

    dummy_df = eda_df if year==[] else eda_df.loc[eda_df.year.isin(year)]

    subj = [] if len(subj)==0 else '|'.join(subj)
    dummy_df = dummy_df if subj==[] else dummy_df.loc[dummy_df['subject'].str.contains(subj)]

    country = [] if len(country)==0 else '|'.join([i for i in country if i!='all'])
    dummy_df = dummy_df if country==[] else dummy_df.loc[dummy_df['affiliation_country'].str.contains(country)]

    st.header("LDA Topic modelling")

    # if scopus and not arxiv:
    #     p = open("lda_scopus.html", 'r')
    # elif arxiv and not scopus:
    #     p = open("lda_arxiv.html", 'r')
    # else:
    p = open("lda_12.html", 'r')
    components.html(p.read(), width=1300, height=800, scrolling=False)

    st.header("Distribution of papers in 2d space based on semantic similarity")
    tsne = eda_df.loc[eda_df['scopus_id'].isin(dummy_df.scopus_id)]
    # q0, q1 = st.columns((0.6, 0.4))
    key_word = st.text_input('Search a keyword ')
    tsne = tsne.loc[tsne['clean'].str.contains(key_word)]
    st.sidebar.metric(label ='Total Research Papers Displayed',value = int(len(tsne)), delta = '', delta_color = 'inverse')
    impact = st.sidebar.checkbox("Show Impact (Scopus)", value=True)
    if arxiv and not scopus:
        fig = px.scatter(data_frame=tsne, x='x0', y='x1',  hover_data=['scopus_id', 'title', 'timestamp', 
                    'subject', 'prism_url', 'year', 'cited_by', 'affilname', 'affiliation_country'], color='c')
    else:
        if impact:
            fig = px.scatter(data_frame=tsne, x='x0', y='x1',  hover_data=['scopus_id', 'title', 'timestamp', 
                    'subject', 'cited_by', 'prism_url', 'year', 'affilname', 'affiliation_country'], color='c', size='cited_by')
        else:
            fig = px.scatter(data_frame=tsne, x='x0', y='x1',  hover_data=['scopus_id', 'title', 'timestamp', 
                    'subject', 'cited_by', 'prism_url', 'year', 'affilname', 'affiliation_country'], color='c')
    legend_names = {tsne.loc[tsne['cluster']==i].iloc[0]['c']:'C-'+str(i) for i in tsne.cluster.unique()}
    fig.for_each_trace(lambda t: t.update(name = legend_names[t.name],
                                    legendgroup = legend_names[t.name],
                                    hovertemplate = t.hovertemplate.replace(t.name, legend_names[t.name])
                                    ))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Description:- ")
    st.text('In-depth analysis using Topic Modelling\n-The first plot is topic visualisation from LDA model \n-The scatter plot is a 2d representation of each research paper')
    st.text(f"From Text to 2D points:- \n1.Title and Abstract of each paper is combined \n2.Converted to vectors using s-BERT model \n3.K-Means for clustering similar papers \n4.T-SNE for dimensionality reduction")




