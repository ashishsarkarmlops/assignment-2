import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st 
from functions import find_target #Function for finding target
from functions import analyse_catagorical_col  #function for analyzing ane categorical column that we want 
from functions import heat_map_coor_plot #Function to perform correlation heatmap analysis 
from functions import plot_null_values #Function to check null values
from functions import replace_yes_no #Function to replace irregularities in data 
from functions import rep_null_val #Function to replace null values
from functions import drop_columns #Function to drop columns 
from functions import cleaning_pipeline #Function that includes cleaning replacing null values and dropping unnecessary columns
###############################################################################################################################
#importing data 
train_df=pd.read_csv('train.csv')
##
st.title('Exploratory Data Analysis') 
st.write('This application is a Streamlit dashboard to explore the dataset.') 
st.dataframe(train_df) # Show the dataset in a dataframe
# Show summary statistics of the dataset 
if st.checkbox('Show summary statistics'):  
    st.write(train_df.describe())
# Show number of missing values in each column  
if st.checkbox('Show number of missing values'):  
    st.write(train_df.isnull().sum())
# Show correlation matrix  
if st.checkbox('Show correlation matrix'):  
    corr_matrix = train_df.corr()  
    st.write(corr_matrix)
# Show distribution plot for each numerical column  
if st.checkbox('Show distribution plot for numerical columns'):
    num_cols = train_df._get_numeric_data().columns 
    for col in num_cols:    

        sns.distplot(train_df[col])    

        plt.title(col)    

        plt.show()