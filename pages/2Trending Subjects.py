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
nltk.download('punkt')
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
            other += i + ','
            other_count += val
    out[other] = other_count
    return out

#this is the header
st.markdown("<h1 style='text-align: center; color: grey;'>Trending Subjects and Key words</h1>", unsafe_allow_html=True)
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

with st.spinner('Updating Report & Generating Word Cloud...'):
    eda_df['affiliation_country'].fillna('Missing Info', inplace=True)
    eda_df['type'].fillna('Missing Info', inplace=True)
    eda_df['scopus_info'].fillna('Missing Info', inplace=True)
    eda_df['cited_by'].fillna(0, inplace=True)

    countries, all_c = eda_from_feature('affiliation_country', eda_df, 300)
    subjects, allSub = eda_from_feature('subject', eda_df)

    country_df = countries.loc[countries['affiliation_country']!='Missing Info']

    s0, s1, s2, s3, s4 = st.columns((0.2, 0.2, 0.2, 0.2, 0.2))
    # Pie chart for Subject
    subj = st.sidebar.multiselect('Choose Subject', subjects.subject.values.tolist(), default='COMP', help = 'Filter report to show info on subject level')
    year = st.sidebar.multiselect('Choose Year', eda_df.year.unique().tolist(), default=[2022, 2021], help = 'Filter report with year')
    country = st.sidebar.multiselect('Choose Country', countries.affiliation_country.values.tolist(), default=[], help = 'Filter report to show info on country level')
    threshold = st.sidebar.number_input('Choose Minimum number of papers', min_value=1, value=100, help = 'Enter number of minimum papers')

    dummy_df = eda_df.loc[eda_df.year.isin(year)]

    subj = subj[0] if len(subj)==0 else '|'.join(subj)
    country = 'all' if country==[] else '|'.join([i for i in country if i!='all'])

    g2, g5 = st.columns((1, 1))
    if country=='all':
        subj_country = dummy_df.loc[dummy_df['subject'].str.contains(subj)]
    else:
        subj_country = dummy_df.loc[dummy_df['affiliation_country'].str.contains(country) 
                                & dummy_df['subject'].str.contains(subj)]
                                
    st.sidebar.metric(label ='Total Research Papers Displayed',value = int(len(subj_country)), delta = '', delta_color = 'inverse')
    dat = breakDown(subj, subjects.subject.values.tolist(), subj_country)
    fig = px.pie(values=dat.values(), names=dat.keys())
    fig.update_layout(title_text=f"Interdisciplinary Research in {subj}",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g2.plotly_chart(fig, use_container_width=True)

    # Countries for Subjects
    # dat = dummy_df.loc[dummy_df['subject'].str.contains(subj)]
    s_countries, all_s_f = eda_from_feature('affiliation_country', subj_country, threshold)
    st.sidebar.metric(label ='Missing Country Info',value = s_countries.iloc[0]['counts'], delta = '', delta_color = 'inverse')
    s_countries_df = s_countries.loc[s_countries['affiliation_country']!='Missing Info']
    # fig = go.Figure(data=[go.Pie(values=s_countries['counts'], labels=s_countries['affiliation_country'], hole=.3)])
    fig = px.bar(s_countries_df, x='affiliation_country', y='counts', color='counts', text='perc')    
    fig.update_layout(title_text=f"Affiliated countries for {subj}",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g5.plotly_chart(fig, use_container_width=True)


    #Word Cloud
    st.write(f"Frequent keywords in {subj}")
    g3 = st.columns((1))[0] 
    fig = make_word_cloud(subj, dummy_df, True, False)
    g3.pyplot(fig)

    st.subheader("Description:- ")
    st.text('Find trending subjects and frequent keywords here. \n-The first plot depicts the interdisciplinary research in each subject \n-The bar chart shows the number of papers affiliated to countries. \n-The word cloud displays frequent keywords from Title and abstract.')




