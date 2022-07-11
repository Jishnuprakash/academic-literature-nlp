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
from nltk import word_tokenize
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')


st.set_page_config(page_title='Analysis on Research Papers',  layout='wide', page_icon=':computer:')
st.set_option('deprecation.showPyplotGlobalUse', False)

#this is the header
 

# t2.title("Analysis on Academic Writing using NLP")
st.image('images/index.jpg', width = 200)
st.markdown("<h1 style='text-align: center; color: grey;'>Analysis on Academic Writing using NLP</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center''> website:<a href='https://virtualengineeringcentre.com'>virtualengineeringcentre.com </a> | email: <a href='mailto:vec@liv.ac.uk'>mailto:vec@liv.ac.uk</a> </div>", unsafe_allow_html=True)
st.text('')

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
    all_f = ",".join(comb_df[feature].dropna()).split(",")
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

## Data

with st.spinner('Updating Report...'):
    
    comb_df = pd.read_csv('combinedDf.csv')

    countries, all_c = eda_from_feature('affiliation_country', comb_df)

    m1, m_0, m2, m_1, m3 = st.columns((1,1,1,1,1))
    m1.metric(label ='Total Research Papers from 2021-22',value = int(len(comb_df)), delta = '', delta_color = 'inverse')
    m2.metric(label ='Total affiliated countries',value = int(len(countries)), delta = '', delta_color = 'inverse')
    m3.metric(label ='Articles Affiliated to United Kingdom',value =int(countries.loc[countries['affiliation_country']=='United Kingdom'].counts) , delta = '', delta_color = 'inverse')

    # Distribution of subjects
    g1 = st.columns((1))[0]
    subjects, allSub = eda_from_feature('subject', comb_df)
    fig = px.bar(subjects, x='subject', y='counts', color='counts', text='perc')    
    fig.update_layout(title_text="Distribution of subjects",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g1.plotly_chart(fig, use_container_width=True)

     # Distribution of Countries
    g0 = st.columns((1))[0]
    fig = px.bar(countries, x='affiliation_country', y='counts', color='counts', text='perc')    
    fig.update_layout(title_text="Affiliated countries",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g0.plotly_chart(fig, use_container_width=True)
    
    # Pie chart for Subject
    subj = st.selectbox('Choose Subject', subjects.subject.values.tolist(), index=0, help = 'Filter report to show only one subject')

    g2, g5 = st.columns((1, 1))
    dat = breakDown(subj, subjects.subject.values.tolist(), comb_df)
    fig = px.pie(values=dat.values(), names=dat.keys())
    fig.update_layout(title_text=f"Interdisciplinary Research in {subj}",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g2.plotly_chart(fig, use_container_width=True)

    # Countries for Subjects
    dat = comb_df.loc[comb_df['subject'].str.contains(subj)]
    s_countries, all_s_f = eda_from_feature('affiliation_country', dat)
    fig = px.bar(s_countries, x='affiliation_country', y='counts', color='counts', text='perc')    
    fig.update_layout(title_text=f"Affiliated countries for {subj}",title_x=0,margin= dict(l=0,r=10,b=10,t=30))
    g5.plotly_chart(fig, use_container_width=True)


    #Word Cloud
    st.write(f"Hot topics in {subj}")
    g3 = st.columns((1))[0] 
    fig = make_word_cloud(subj, comb_df, True, False)
    g3.pyplot(fig)


with st.spinner('Updating Scatterplot...'):
    # Clustering
    st.header("Distribution of papers in 2d space based on their semantic similarity")

    tsne = pd.read_csv("tsneDF_15.csv")
    with open('topics15.txt') as f:
        topics = f.readlines()

    cluster = st.slider("Select a Cluster Number", min_value=0, max_value=len(topics), value=len(topics), step=1)

    if cluster==len(topics):
        dat = tsne
        text = " ".join(topics)
        heading = f"Topics generated from all research papers"
    else:
        dat = tsne.loc[tsne['cluster']==int(cluster)]
        text = topics[int(cluster)]
        heading = f"Topics generated from cluster {cluster}"
        
    
    fig = px.scatter(data_frame=dat, x='x0', y='x1',  hover_data=['scopus_id', 'title', 'timestamp', 
                     'subject'], width=1200, height=1500, color='cluster' )
    fig.update_layout(showlegend=True)
    st.write(f"Title and Abstract of each paper is combined and converted to vectors using s-BERT model, k-Means for clustering similar papers, & t-SNE for dimensionality reduction")
    st.plotly_chart(fig, use_container_width=True)
    st.write(heading)
    fig = gen_wordcloud(text)
    st.pyplot(fig)
    # if cluster!=len(topics):
    #     st.text(f"Key words in cluster {cluster}, are {topics[cluster]}\n\n These are found by Latent Dirichlet allocation - Topic modelling")


# Contact Form
with st.expander("Contact us"):
    with st.form(key='contact', clear_on_submit=True):
        
        email = st.text_input('Contact Email')
        st.text_area("Query","Please fill in all the information or we may not be able to process your request")  
        
        submit_button = st.form_submit_button(label='Send Information')
        
        
        
        
        
        
        
        
        
        