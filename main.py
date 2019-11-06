from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import feature_selection as f_select
import os.path
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
# from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import requests
from bs4 import BeautifulSoup
import lxml
from lxml import html
import string
import time


### GEt YEAR
strings = time.strftime("%Y,%m,%d,%H,%M,%S")
t = strings.split(',')
numbers = [ int(x) for x in t ]
year=numbers[0]

print("App Starting/Gathering Data....Please Wait until URL provided")


#This takes the player stats data, and creates a list of a lists, where a list is all the values of a player
url = 'https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_per_game.html'
response = requests.get(url)
page = response.text
soup = BeautifulSoup(page,"lxml")
results = []

for row in soup.find_all(class_="full_table")[1:]:
    data = row.find_all('td')
    Name = data[0].text
    Pos = data[1].text
    Age = data[2].text
    Tm = data[3].text
    G = data[4].text
    GS = data[5].text
    MP = data[6].text
    FG = data[7].text
    FGA = data[8].text
    FG_Per = data[9].text
    threeP = data[10].text
    threePA = data[11].text
    threeP_Perc = data[12].text
    twoP = data[13].text
    twoPA = data[14].text
    twoP_Per = data[15].text
    eFG_Per = data[16].text
    FT = data[17].text
    FTA = data[18].text
    FT_Per = data[19].text
    ORB = data[20].text
    DRB = data[21].text
    TRB = data[22].text
    AST = data[23].text
    STL = data[24].text
    BLK = data[25].text
    TOV = data[26].text
    PF = data[27].text
    PS_Game = data[28].text
    # get the rest of the data you need about each coin here, then add it to the dictionary that you append to results
    results.append([Name, Pos, Age, Tm, G, GS, MP, FG, FGA, FG_Per, threeP, \
    threePA, threeP_Perc, twoP, twoPA, twoP_Per, eFG_Per, FT, \
    FTA, FT_Per, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PS_Game])

#create a pandas dataframe
pd.set_option('display.max_columns', 500)
players_df = pd.DataFrame(results, columns=['name', 'Pos', 'Age',\
 'Tm', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per', \
'threeP', 'threePA', 'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per', \
'eFG_Per', 'FT', 'FTA', 'FT_Per', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS_Game'])

# #replace missing values with 0
for i in range(len(players_df)) :
    for j in range(29):
        if players_df.iloc[i,j]=="":
            players_df.iloc[i,j]=0


### Scrape Salary
salaries_url = 'https://www.basketball-reference.com/contracts/players.html'
salaries_response = requests.get(salaries_url)
page = salaries_response.text
soup = BeautifulSoup(page,"lxml")
tree = lxml.html.fromstring(page)
i_need_element = tree.xpath('//*[@id="content"]/div[2]/p[1]/text()[1]')
market_cap=i_need_element[0].split(" ")[1][1:].replace(",","")
market_cap=int(market_cap)
salaries = []
table_body=soup.find_all('tbody')
for tb in table_body:
    x =tb.find_all('tr')
    for row in x:
        try:
            tds_salaries = row.find_all('td')
            name_s = tds_salaries[0].text
            team_e = tds_salaries[1].text
            salary = tds_salaries[2].text
            salaries.append([name_s,team_e, salary[1:]])
            cnt+=1
        except:
            pass
#create a salary pandas dataframe
salaries_df = pd.DataFrame(salaries, columns=['name','te', 'salary'])
#take out all the commas in salary column
# salaries_df['salary'] = salaries_df['salary'].replace(',', '')
for i in range(len(salaries_df['salary'])):
    salaries_df['salary'][i]=salaries_df['salary'][i].replace(",","")
#there are duplicate rows for people that got traded..so we need to groupby and calculate their average salary!
#convert salary column to integers - so I can take the mean salaries of the players that got traded-remove duplicates!!
salaries_df['salary'] = pd.to_numeric(salaries_df['salary'], errors='coerce', downcast='float').fillna(0)
#then merge the 2 dataframes on column "Name" - no one in the NBA with the same name
nba_df = pd.merge(players_df, salaries_df, how='left', on='name')
#drop all the rows with NaN values for 'salary' -- either they retired or did not make a team this year
nba_df = nba_df.dropna()
#convert the numerial columns to floats
nba_df[['Age','G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per', 'threeP', 'threePA', 'threeP_Perc', \
'twoP', 'twoPA', 'twoP_Per', 'eFG_Per', 'FT', 'FTA', 'FT_Per', 'ORB', 'DRB', 'TRB', 'AST', 'STL',\
 'BLK', 'TOV', 'PF', 'PS_Game']] = nba_df[['Age','G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per', 'threeP', 'threePA',\
  'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per', 'eFG_Per', 'FT', 'FTA', 'FT_Per', \
  'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS_Game']].apply(pd.to_numeric, errors='coerce')
#Rename 'mean' column to 'salary'
nba_df = nba_df.rename(columns = {'mean':'salary'})
nba_df['Tm']=nba_df['te']
nba_df.drop(['te'], axis = 1, inplace=True)
#create playoff column
nba_df['playoff'] = 0
## Play Playoffs
results = []
years=['2018','2019']
for y in range(2):
    url = 'https://www.basketball-reference.com/playoffs/NBA_'+years[y]+'_totals.html'
    response = requests.get(url)
    page = response.text
    soup = BeautifulSoup(page,"lxml")
    for row in soup.find_all(class_="full_table")[1:]:
        data = row.find_all('td')
        play_off_tm = data[3].text
        # get the rest of the data you need about each coin here, then add it to the dictionary that you append to results
        results.append(play_off_tm)

output = []
for x in results:
    if x not in output:
        output.append(x)
for x in output:
    nba_df.loc[nba_df['Tm'] == x, "playoff"] = 1
#save my dataset as a csv file
# nba_df.to_csv('nba_FINALDATA.csv', sep=',')
## salary as sqrt(y)
nba_df['sqrty'] = np.sqrt(nba_df.salary)
#save my dataset as a csv file
# nba_df.to_csv('nba_FINALDATA_ytransform.csv', sep=',')

### REGRESSION MODEL ####

#splitting it into 80/20 - smaller dataset
X_train, X_test, y_train, y_test = train_test_split(nba_df.loc[:, nba_df.columns != 'sqrty'], \
nba_df.sqrty, test_size=0.2, random_state =1234)
#train
y = y_train
X = X_train[['G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per','threeP', 'threePA', \
'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per','eFG_Per', 'FT', 'FTA', 'FT_Per', \
'ORB', 'DRB', 'TRB', 'AST', 'STL','BLK', 'TOV', 'PF', 'PS_Game', 'playoff']]
#test
y_test = y_test
X_test = X_test[['G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per','threeP', 'threePA', \
'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per','eFG_Per', 'FT', 'FTA', 'FT_Per',\
 'ORB', 'DRB', 'TRB', 'AST', 'STL','BLK', 'TOV', 'PF', 'PS_Game', 'playoff']]

##Ridge Linear Regression Model
##find the best hyperparameters using grid search##
# Define the parameter values that should be searched
alpha_range = [1e-4, 1e-3, 1.5e-3, 1e-2, 1.5e-2, 1e-1, 1, 5, 10, 50, 100, 1000, 10000, 100000]
normalize_range = [True, False]
# Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(alpha=alpha_range, normalize=normalize_range)
# Instantiate the grid
grid = GridSearchCV(Ridge(), param_grid, cv=5)
grid.fit(X, y)
ridge = Ridge(alpha=1, normalize=False)
# Fit model
ridge.fit(X, y)

player_data=nba_df.loc[:, nba_df.columns != 'sqrty']
player_pred=player_data[['G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per','threeP', 'threePA', \
'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per','eFG_Per', 'FT', 'FTA', 'FT_Per',\
 'ORB', 'DRB', 'TRB', 'AST', 'STL','BLK', 'TOV', 'PF', 'PS_Game', 'playoff']]
y_pred=ridge.predict(player_pred)

# y_pred_test = ridge.predict(X_test)
# #test data
# plt.scatter(y_test**2, y_pred_test**2)
# plt.title("Actual Salary Vs. Predicted Salary")
# plt.xlabel("Actual Salary")
# plt.ylabel("Predicted Salary")
# plt.show()
#
#
#
# player_data=nba_df.loc[nba_df['name'] == 'Stephen Curry']
# player_pred=player_data[['G', 'GS', 'MP', 'FG', 'FGA', 'FG_Per','threeP', 'threePA', \
# 'threeP_Perc', 'twoP', 'twoPA', 'twoP_Per','eFG_Per', 'FT', 'FTA', 'FT_Per',\
#  'ORB', 'DRB', 'TRB', 'AST', 'STL','BLK', 'TOV', 'PF', 'PS_Game', 'playoff']]
#
# y_pred = ridge.predict(player_pred)
# print(player_data['salary'],y_pred**2)

show_df=nba_df[['name','Pos','Age','Tm','salary']]
show_df['Predicted_Salary']=y_pred**2
show_df['EPP']=((y_pred**2-nba_df['salary'])/nba_df['salary'])*100
show_df=pd.concat([show_df, player_pred], axis=1, sort=False)
show_df=show_df.round(2)
show_df=show_df.sort_values(by='salary', ascending=False)
make_float = lambda x: "${:,.2f}".format(x)
make_per = lambda x: "{:,.2f}%".format(x)
show_df['Predicted_Salary']=show_df['Predicted_Salary'].apply(make_float)
show_df['salary']=show_df['salary'].apply(make_float)
show_df['EPP']=show_df['EPP'].apply(make_per)
market_cap=float(market_cap)
market_cap=make_float(market_cap)

## Styling
def color_negative_red(value):
  """
  Colors elements in a dateframe
  green if positive and red if
  negative. Does not color NaN
  values.
  """

  if value < 0:
    color = 'red'
  elif value > 0:
    color = 'green'
  else:
    color = 'black'

  return 'color: %s' % color


show_df.style.applymap(color_negative_red, subset=['salary','Predicted_Salary'])

# ## subplots
# # money spend by team
# team_sals = nba_df.groupby("Tm").agg(np.sum).sort_values('salary', ascending=False).reset_index()
# fig, ax = plt.subplots(figsize=(12,5))
# sns.barplot(x=team_sals.sqrty, y=team_sals.Tm, ax=ax)
# ax.set_xlabel('Salary Total for this year')
# ax.set_title('Salary Comparision by Team')
# plt.savefig('./static/infographics/avgsal.png')

app = Flask(__name__)
# app._static_folder = '\static'
@app.route('/', methods=("POST", "GET"))
def html_table():

    # return render_template('pred_table.html',  tables=[show_df.to_html(classes='data')], titles=show_df.columns.values)
    return render_template("pred_table.html", market=market_cap,column_names=show_df.columns.values, row_data=list(show_df.values.tolist()), zip=zip)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000)

# app = Flask(__name__)
#
# @app.route('/', methods=['post', 'get'])
# def login():
#     message = ''
#     if request.method == 'POST':
#         username = request.form.get('username')  # access the data inside
#         password = request.form.get('password')
#
#         if username == 'root' and password == 'pass':
#             message = "Correct username and password"
#         else:
#             message = "Wrong username or password"
#
#     return render_template('pred.html', message=message)
#
#
# # main driver function
# if __name__ == '__main__':
#     app.run()
