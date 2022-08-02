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
            other += i + ' '
            other_count += val
    out[other] = other_count
    return out

#this is the header
st.image('images/index.jpg', width=200)
st.markdown("<h1 style='text-align: center; color: black;'>Research Mapping Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>(An analysis on academic literature using NLP)</h3>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center''> website:<a href='https://virtualengineeringcentre.com'>virtualengineeringcentre.com </a> | email: <a href='mailto:vec@liv.ac.uk'>mailto:vec@liv.ac.uk</a> </div>", unsafe_allow_html=True)
st.text('')

data = st.sidebar.multiselect('Choose API', ['Scopus', 'Arxiv'], default=['Scopus', 'Arxiv'], help = 'Filter report to show results based on API')

# data = st.multiselect('Choose API', ['Scopus', 'Arxiv'], default=['Scopus', 'Arxiv'], help = 'Filter report to show results based on API')
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

    m1, m2, m_1, m3 = st.columns((1,1,1,1))
    m1.metric(label =f"Total Research Papers from {min(eda_df.year)} to {max(eda_df.year)}",value = int(len(eda_df)), delta = '', delta_color = 'inverse')
    m2.metric(label ='Total affiliated countries',value = int(len(countries))-1, delta = '', delta_color = 'inverse')
    m_1.metric(label ='Missing country information',value = countries.iloc[0]['counts'], delta = '', delta_color = 'inverse')
    uk_papers = int(countries.loc[countries['affiliation_country']=='United Kingdom'].counts) if 'United Kingdom' in " ".join(countries['affiliation_country']) else 0
    m3.metric(label ='Papers Affiliated to United Kingdom', value=uk_papers, delta = '', delta_color = 'inverse')

    
    # Distribution of subjects
    g0, g1 = st.columns((1, 1))
    subjects, allSub = eda_from_feature('subject', eda_df)
    fig = px.bar(subjects, x='subject', y='counts', color='counts', text='perc')    
    fig.update_layout(title_text="Distribution of subjects",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g0.plotly_chart(fig, use_container_width=True)

    #Distribution of Years
    years, allYear = eda_from_feature('year', eda_df)
    years.sort_values(by='year', ascending=False, inplace=True)
    fig = px.pie(years, names='year', values='counts', color='counts') 
    fig.update_layout(title_text="Distribution of Year",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g1.plotly_chart(fig, use_container_width=True)


    # Distribution of Countries
    country_df = countries.loc[countries['affiliation_country']!='Missing Info']
    if len(country_df)>0:
        c0 = st.columns((1))[0]
        fig = px.bar(country_df, x='affiliation_country', y='counts', color='counts', text='perc')
        fig.update_layout(title_text="Affiliated countries",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
        c0.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Description:- ")
    st.text('This is an overview of data (research papers) collected from Scopus and Arxiv API. \n-The first plot depicts a distribution of subjects. \n-The Pie chart shows paper per year \n-The final barchart represents the country affiliations.')
    st.text('Note: Arxiv API does not have country information.')

            
# df = base_df[['scopus_id', 'title', 'timestamp', 'subject', 'clean', 'year', 'api_flag', 'cluster', 
#               'x0', 'x1', 'c', 'affiliation_country', 'affilname',  'prism_url', 'cited_by']]
# df.to_csv('dash/tsne_scopus_arxiv_small12.csv', index=False)