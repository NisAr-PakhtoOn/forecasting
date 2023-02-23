import pandas as pd
import streamlit as st
from pycaret.time_series import *

#------------------------
# here we are going to set the page layout
st.set_page_config(page_title='Machine Learning App with Random Forest',layout='wide')

st.write("""
# Prediction Model

""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    forcast = st.text_input('Enter how many days you want to forecast')



#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Show the data in the UI
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    st.markdown('**1.1. Glimpse of dataset**')
    st.table(df.head(10))

    # TRAINING THE MODEL HERE

    exp_name = setup(data = df,  fh = 12)
    arima = create_model('gbr_cds_dt')
    pred_holdout = predict_model(arima)
    forecast = predict_model(finalize_model(arima), fh = int(forcast))
    st.text(forecast.to_string())
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_model(arima, plot = 'forecast', data_kwargs = {'fh' : int(forcast)}))

else:
    st.info('Awaiting for CSV file to be uploaded.')
    

