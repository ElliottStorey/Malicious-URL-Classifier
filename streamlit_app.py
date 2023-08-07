from urllib.parse import urlparse, parse_qs
from classifier import predict, vectorizer
import streamlit as st
import torch
import matplotlib.pyplot as plt

def display_url_features(url):
    if not urlparse(url).scheme:
        parsed_url = urlparse(f'//{url}')
    else:
        parsed_url = urlparse(url)
    domain = parsed_url.netloc
    scheme = parsed_url.scheme
    path = parsed_url.path
    query_params = parse_qs(parsed_url.query)

    with st.expander('Show URL Details', True):
        st.markdown(f'**Domain:** {domain}')
        st.markdown(f'**Scheme:** {scheme}')
        st.markdown(f'**Path:** {path}')
        if query_params:
            st.markdown('**Query Parameters:**')
            for param, value in query_params.items():
                st.markdown(f'- {param}: {value[0]}')

def plot_vector(vector, label):
    plt.figure(figsize=(8, 4))
    dimensions = [f'Dim {i+1}' for i in range(len(vector))]
    plt.bar(dimensions, vector)
    plt.xlabel('')
    plt.ylabel('Value')
    plt.title('URL Vector Representation')
    plt.grid(axis='y')
    plt.gca().set_xticklabels([])  # Remove x-axis tick labels
    plt.tight_layout()
    st.pyplot(plt)

def display_prediction(prediction):
    if prediction > 0.75:
        st.warning('🚨 This URL is most likely malicious.')
    elif prediction < 0.25:
        st.success('✅ This URL is most likely safe.')
    else:
        st.info('🤔 The URL is ambiguous, cannot determine its safety with high confidence.')

    st.progress(prediction)

def main():
    st.set_page_config(page_title='Malicious URL Classifier')
    st.title('Malicious URL Classifier')
    st.write('This is a web app that classifies the safety of a given URL. The classification model was trained on over 800,000 URLs labeled as either malicious or non-malicious. Enter a URL in the text box below to see the prediction.')
    input_url = st.text_input('Enter the URL to classify:')
    if input_url:
        st.write('')
        st.write('Classifying the URL...')
        with st.spinner('Please wait...'):
            prediction = predict(input_url)
        st.write('')
        display_prediction(prediction)
        st.write('')
        st.subheader('URL Features')
        display_url_features(input_url)
        st.write('')
        st.subheader('URL Vector')
        with st.spinner('Loading URL vector...'):
            url_vector = vectorizer(input_url)
        
        if url_vector is not None:
            plot_vector(url_vector, input_url)

    st.write('---')
    st.write('Made by Elliott Storey @ COLSA')

if __name__ == '__main__':
    main()