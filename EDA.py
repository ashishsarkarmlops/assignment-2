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
income_train_df=pd.read_csv('train.csv')
###############################################################################################################################
#Separating columns in different data types 
float_columns=income_train_df.select_dtypes('float').columns.tolist()
int_columns=income_train_df.select_dtypes('int').columns.tolist()
object_columns=income_train_df.select_dtypes('object').columns.tolist()
###############################################################################################################################
st.title('Description') 
st.write('Identify the level of income qualification needed for the families in Latin America')
st.title('Problem Statement')
st.write("""Many social programs have a hard time making sure the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of population can’t provide the necessary income and expen|se records to prove that they qualify.

In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to classify them and predict their level of need. While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

The Inter-American Development Bank (IDB) believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.""")
st.write('This application is a Streamlit dashboard to explore the dataset.') 
if st.checkbox('Show Data Dictionary'): 
    """ 
1.	ID = Unique ID
2.	v2a1, Monthly rent payment
3.	hacdor, =1 Overcrowding by bedrooms
4.	rooms, number of all rooms in the house
5.	hacapo, =1 Overcrowding by rooms
6.	v14a, =1 has bathroom in the household
7.	refrig, =1 if the household has a refrigerator
8.	v18q, owns a tablet
9.	v18q1, number of tablets household owns
10.	r4h1, Males younger than 12 years of age
11.	r4h2, Males 12 years of age and older
12.	r4h3, Total males in the household
13.	r4m1, Females younger than 12 years of age
14.	r4m2, Females 12 years of age and older
15.	r4m3, Total females in the household
16.	r4t1, persons younger than 12 years of age
17.	r4t2, persons 12 years of age and older
18.	r4t3, Total persons in the household
19.	tamhog, size of the household
20.	tamviv, number of persons living in the household
21.	escolari, years of schooling
22.	rez_esc, Years behind in school
23.	hhsize, household size
24.	paredblolad, =1 if predominant material on the outside wall is block or brick
25.	paredzocalo, "=1 if predominant material on the outside wall is socket (wood, zinc or absbesto"
26.	paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
27.	pareddes, =1 if predominant material on the outside wall is waste material
28.	paredmad, =1 if predominant material on the outside wall is wood
29.	paredzinc, =1 if predominant material on the outside wall is zink
30.	paredfibras, =1 if predominant material on the outside wall is natural fibers
31.	paredother, =1 if predominant material on the outside wall is other
32.	pisomoscer, "=1 if predominant material on the floor is mosaic, ceramic, terrazo"
33.	pisocemento, =1 if predominant material on the floor is cement
34.	pisoother, =1 if predominant material on the floor is other
35.	pisonatur, =1 if predominant material on the floor is natural material
36.	pisonotiene, =1 if no floor at the household
37.	pisomadera, =1 if predominant material on the floor is wood
38.	techozinc, =1 if predominant material on the roof is metal foil or zink
39.	techoentrepiso, "=1 if predominant material on the roof is fiber cement, mezzanine "
40.	techocane, =1 if predominant material on the roof is natural fibers
41.	techootro, =1 if predominant material on the roof is other
42.	cielorazo, =1 if the house has ceiling
43.	abastaguadentro, =1 if water provision inside the dwelling
44.	abastaguafuera, =1 if water provision outside the dwelling
45.	abastaguano, =1 if no water provision
46.	public, "=1 electricity from CNFL, ICE, ESPH/JASEC"
47.	planpri, =1 electricity from private plant
48.	noelec, =1 no electricity in the dwelling
49.	coopele, =1 electricity from cooperative
50.	sanitario1, =1 no toilet in the dwelling
51.	sanitario2, =1 toilet connected to sewer or cesspool
52.	sanitario3, =1 toilet connected to septic tank
53.	sanitario5, =1 toilet connected to black hole or letrine
54.	sanitario6, =1 toilet connected to other system
55.	energcocinar1, =1 no main source of energy used for cooking (no kitchen)
56.	energcocinar2, =1 main source of energy used for cooking electricity
57.	energcocinar3, =1 main source of energy used for cooking gas
58.	energcocinar4, =1 main source of energy used for cooking wood charcoal
59.	elimbasu1, =1 if rubbish disposal mainly by tanker truck
60.	elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
61.	elimbasu3, =1 if rubbish disposal mainly by burning
62.	elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
63.	elimbasu5, "=1 if rubbish disposal mainly by throwing in river, creek or sea"
64.	elimbasu6, =1 if rubbish disposal mainly other
65.	epared1, =1 if walls are bad
66.	epared2, =1 if walls are regular
67.	epared3, =1 if walls are good
68.	etecho1, =1 if roof are bad
69.	etecho2, =1 if roof are regular
70.	etecho3, =1 if roof are good
71.	eviv1, =1 if floor are bad
72.	eviv2, =1 if floor are regular
73.	eviv3, =1 if floor are good
74.	dis, =1 if disable person
75.	male, =1 if male
76.	female, =1 if female
77.	estadocivil1, =1 if less than 10 years old
78.	estadocivil2, =1 if free or coupled uunion
79.	estadocivil3, =1 if married
80.	estadocivil4, =1 if divorced
81.	estadocivil5, =1 if separated
82.	estadocivil6, =1 if widow/er
83.	estadocivil7, =1 if single
84.	parentesco1, =1 if household head
85.	parentesco2, =1 if spouse/partner
86.	parentesco3, =1 if son/doughter
87.	parentesco4, =1 if stepson/doughter
88.	parentesco5, =1 if son/doughter in law
89.	parentesco6, =1 if grandson/doughter
90.	parentesco7, =1 if mother/father
91.	parentesco8, =1 if father/mother in law
92.	parentesco9, =1 if brother/sister
93.	parentesco10, =1 if brother/sister in law
94.	parentesco11, =1 if other family member
95.	parentesco12, =1 if other non family member
96.	idhogar, Household level identifier
97.	hogar_nin, Number of children 0 to 19 in household
98.	hogar_adul, Number of adults in household
99.	hogar_mayor, # of individuals 65+ in the household
100.	hogar_total, # of total individuals in the household
101.	dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
102.	edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
103.	edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
104.	meaneduc,average years of education for adults (18+)
105.	instlevel1, =1 no level of education
106.	instlevel2, =1 incomplete primary
107.	instlevel3, =1 complete primary
108.	instlevel4, =1 incomplete academic secondary level
109.	instlevel5, =1 complete academic secondary level
110.	instlevel6, =1 incomplete technical secondary level
111.	instlevel7, =1 complete technical secondary level
112.	instlevel8, =1 undergraduate and higher education
113.	instlevel9, =1 postgraduate higher education
114.	bedrooms, number of bedrooms
115.	overcrowding, # persons per room
116.	tipovivi1, =1 own and fully paid house
117.	tipovivi2, "=1 own, paying in installments"
118.	tipovivi3, =1 rented
119.	tipovivi4, =1 precarious
120.	tipovivi5, "=1 other(assigned, borrowed)"
121.	computer, =1 if the household has notebook or desktop computer
122.	television, =1 if the household has TV
123.	mobilephone, =1 if mobile phone
124.	qmobilephone, # of mobile phones
125.	lugar1, =1 region Central 
126.	lugar2, =1 region Chorotega
127.	lugar3, =1 region PacÃfico central
128.	lugar4, =1 region Brunca
129.	lugar5, =1 region Huetar AtlÃ¡ntica
130.	lugar6, =1 region Huetar Norte
131.	area1, =1 zoinformationna urbana
132.	area2, =2 zona rural
133.	age= Age in years
134.	SQBescolari= escolari squared
135.	SQBage, age squared
136.	SQBhogar_total, hogar_total squared
137.	SQBedjefe, edjefe squared
138.	SQBhogar_nin, hogar_nin squared
139.	SQBovercrowding, overcrowding squared
140.	SQBdependency, dependency squared
141.	SQBmeaned, square of the mean years of education of adults (>=18) in the household
142.	agesq= Age squared
    """
if st.checkbox('Show sample dataset'):
    st.write('Sample Dataset') 
    st.dataframe(income_train_df.iloc[:5]) # Show the dataset in a dataframe
#Show data size column types 
if st.checkbox('Show columns and dtype information'):
    st.write(f'Float type columns => {float_columns}')
    st.write(f'Object type columns => {object_columns}')
    st.write(f'Numerical columns => {int_columns}')
# Show summary statistics of the dataset 
if st.checkbox('Show summary statistics'):  
    st.write(income_train_df.describe())
# Show number of missing values in each column  
if st.checkbox('Show number of missing values'):  
    st.write("Float Columns null value distribution")
    dd=pd.DataFrame(income_train_df[float_columns].isnull().sum()/income_train_df.shape[0]).reset_index()
    dd.columns=['Column Name','Null Val %']
    st.dataframe(dd)
    #st.write(f"{income_train_df[float_columns].isnull().sum()/income_train_df.shape[0]}")
    plt.show()
    st.pyplot()
if st.checkbox('Show number of missing values'):
    st.bar_chart(plot_null_values(income_train_df[float_columns],(4,2)))
# Show correlation matrix  
if st.checkbox('Show correlation matrix'):  
    corr_matrix = income_train_df.corr()  
    st.write(corr_matrix)
# Show distribution plot for each numerical column  
if st.checkbox('Show distribution plot for numerical columns'):
    num_cols = income_train_df._get_numeric_data().columns 
    for col in num_cols:    

        sns.distplot(income_train_df[col])    

        plt.title(col)    

        plt.show()