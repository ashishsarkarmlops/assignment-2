import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

"""This File contains the reposiitory of all the function being used in the Income qualification project""" 
def Read_data(filename):
    #this function reads the file from the current directory
    return pd.read_csv(filename)

def find_target(df1,df2):
    """Will be used to find the target variable from the data sets 
    #function takes two datasets as input and provides with the column name that is not present in the test set""" 
    return (list(set(df1.columns)-set(df2.columns)))[0]
def analyse_catagorical_col(df,col_name,figsize):
    """Function can be used to analyse the categorical features would provide a countplot as an output for the provided (datafarame and column name and a tuple for figuresize) are the expected input variables  
    to adjust figsize""" 
    plt.figure(figsize=figsize)
    """forms a countplot for each category"""
    sns.countplot(x=df[f'{col_name}'])
    """Set the labels and title of the plot"""
    plt.title(f'{col_name} cardinality distribution')
    """Show the plot""" 
    plt.show()
def heat_map_coor_plot(var1,figsize):
    """This function creates a heatmap on the highly correlated features 
    takes a correlated matrix as one feature and figsize as s tuple"""
    plt.figure(figsize=figsize)
    sns.heatmap(var1,annot=True, cmap = sns.color_palette("Set2", 8), fmt='.3f')
def plot_null_values(df):
    null_values = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
    null_values = pd.DataFrame(null_values)
    """Reset the index to form it in proper dataframe"""
    null_values.reset_index(inplace=True)
    """Renaming columns""" 
    null_values.columns=['Feature','Percentage of missing values']
    return null_values
    
def replace_yes_no(df): 
    """This function replaces the Yes:1 and No:0 for the following columns and returns 
       the dataframe expects the the dataframe as input"""
    mapping = {'yes' :1, 'no' :0}
    for i in df:
        df['dependency']= df['dependency'].replace(mapping).astype(float)
        df['edjefe']= df['edjefe'].replace(mapping).astype(float)
        df['edjefa']= df['edjefa'].replace(mapping).astype(float)
        df['meaneduc']= df['meaneduc'].replace(mapping).astype(float)
    return df
def check_all_member_same_target(df):
    """This function checks in the data with the help if idhogar column as that is the unique 
    identification for each family that the poverty level is same ir not"""
    all_equal=df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
    not_equal = all_equal[all_equal != True]
    return len(not_equal)
def check_with_head_or_not(df):
    """This function check for the number of families that are without heads"""
    households_head = df.groupby('idhogar')['parentesco1'].sum()
    # Find households without a head
    households_no_head = df.loc[df['idhogar'].isin(households_head[households_head == 0].index), :]
    return households_no_head['idhogar'].nunique()
def rep_null_val(df):
    """Replaces all null values the dataframe 
       for following column replacing with 0 as per the finsdings during discovery""" 
    df['v2a1'].fillna(0,inplace=True)
    """For following column replacing with 0 as per the findings during discovery"""
    df['v18q1'].fillna(0,inplace=True)
    """For following column replacing with 0 as per the findings during discovery"""
    df['rez_esc'].fillna(0,inplace=True)
    """For following column replacing with 'edjefe' columns respective value as per the findings during discovery"""
    df['meaneduc'].fillna(df['edjefe'],inplace=True)
    """For following column replacing with square of 'meaneduc' columns respective value as per the findings during discovery"""
    df['SQBmeaned'].fillna(df['meaneduc']**2,inplace=True)
    return df
def drop_columns(df):
    """This function takes the complete dataframe as input drops the unwanted columns and provides the exactly required dataframe""" 
    df.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq','Id','idhogar','coopele', 'area2', 'tamhog', 'hhsize', 'hogar_total', 'r4t3','area2'],axis=1,inplace=True)
    return df 
def cleaning_pipeline(df):
    """Function takes the dataframe then 
        1) replaces yes with 1 and no with 0 
        2) replaces all the null values in the reuired pattern as per discovery
        3) deletes all the unwanted columns"""  
    df1=replace_yes_no(df)
    df2=rep_null_val(df1)
    df3=drop_columns(df2)
    return df3
def problem_stat():
    """DESCRIPTION

Identify the level of income qualification needed for the families in Latin America

# Problem Statement Scenario:
Many social programs have a hard time making sure the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of population can’t provide the necessary income and expen|se records to prove that they qualify.

In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to classify them and predict their level of need. While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

The Inter-American Development Bank (IDB) believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance."""
def steps():
    """
# Following actions should be performed:
* Identify the output variable.
* Understand the type of data.
* Check if there are any biases in your dataset.
* Check whether all members of the house have the same poverty level.
* Check if there is a house without a family head.
* Set the poverty level of the members and the head of the house same in a family.
* Count how many null values are existing in columns.
* Remove null value rows of the target variable.
* Predict the accuracy using random forest classifier.
* Check the accuracy using a random forest with cross-validation."""
def data_dictionary():
    """1.	ID = Unique ID
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
142.	agesq= Age squared"""
def target_inference():
    """From above we can observe that the training data is biased as the model will 
get the opportunity learn very few Extreme poverty cases. And that can lead to a state where the
model not identify the Extreme poor cases at all
    """