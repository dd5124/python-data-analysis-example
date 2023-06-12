'''
Data analysis example
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from doubleml import DoubleMLPLR
from doubleml import DoubleMLData

def import_data(sales: str, store: str):
    '''
    Imports and cleans data
    Returns a list of original datasets and  cleaned merged dataset
    '''
    df1 = pd.read_csv(sales)
    df2 = pd.read_csv(store)
    data_frame = pd.merge(df1,df2,left_on='store', right_on='STORE',how='left')
    data_frame = data_frame.drop(columns=['constant'])
    prices = data_frame.filter(like='price').columns.tolist()
    other_vars = data_frame.columns.difference(prices)
    result = data_frame.melt(id_vars=other_vars, value_vars=prices,value_name='price')
    data_frame = result
    # merge brands and change brand integer label to string
    b_name = {
        'Tropicana': [1, 2, 4],
        'Florida_natural': [3],
        'Minute_maid': [5, 6],
        'Citrus_hill': [7],
        'Tree_fresh': [8],
        'Florida_gold': [9],
        'Dominicks': [10,11]}
    b_size = {
        64: [1, 3, 4, 5, 7, 8, 9, 10],
        96: [2, 6],
        128: [11]}
    data_frame['new_brand'] = data_frame.brand.map({item: k for k,\
        v in b_name.items() for item in v})
    data_frame['new_brand'] = data_frame['new_brand'].astype('string')
    data_frame['size'] = data_frame.brand.map({item: k for k, v in b_size.items() for item in v})
    data_frame = data_frame.drop(['brand', 'variable'],axis=1)
    data_frame = data_frame.rename({'new_brand': 'brand'}, axis=1)

    return data_frame

def data_visualization1(data_frame):
    '''
    Creates data visualization
    Returns graphs
    '''
    temp_data = data_frame.groupby(['brand']).sum().reset_index()
    graph = plt.subplots(2, 1, figsize = (40,25))

    sns.barplot(x='brand', y='profit', data=temp_data, ax=graph[1][0])
    graph[1][0].set_xlabel(graph[1][0].get_xlabel(), size=30)
    graph[1][0].set_ylabel(graph[1][0].get_ylabel(), size=30)
    graph[1][0].set_xticklabels(graph[1][0].get_xticklabels(), size=30)
    graph[1][0].bar_label(graph[1][0].containers[0], fmt='%.2f', size=25)
    graph[1][0].set_title('Total Profit per Brand', size= 40)

    sns.barplot(x='brand', y='logmove', data=temp_data, ax=graph[1][1])
    graph[1][1].set_xlabel(graph[1][1].get_xlabel(), size=30)
    graph[1][1].set_ylabel(graph[1][1].get_ylabel(), size=30)
    graph[1][1].set_xticklabels(graph[1][1].get_xticklabels(), size=30)
    graph[1][1].bar_label(graph[1][1].containers[0], fmt='%.2f', size=25)
    graph[1][1].set_title('Total Quantity Sold per Brand', size= 40)

    return graph

def data_visualization2(data_frame):
    '''
    Creates data visualization
    Returns graphs
    '''
    graph = plt.subplots(2, 1, figsize=(15,10))

    data_frame.groupby('week').sum()[['logmove']].plot(ax=graph[1][0])
    graph[1][0].set_xlabel(graph[1][0].get_xlabel(), size=30)
    graph[1][0].set_ylabel(graph[1][0].get_ylabel(), size=30)
    graph[1][0].set_title('Total Logmove over Week', size= 40)


    data_frame.groupby('week').sum()[['profit']].plot(ax=graph[1][1])
    graph[1][1].set_xlabel(graph[1][1].get_xlabel(), size=30)
    graph[1][1].set_ylabel(graph[1][1].get_ylabel(), size=30)
    graph[1][1].set_title('Total Profit over Week', size= 40)

    graph[0].tight_layout()

    return graph

def descriptive_statistics(data_frame):
    '''
    Generates descriptive statistics from data
    Returns a list of statistics of interest
    '''
    highest_sales = data_frame.groupby('brand').sum()[['logmove', 'profit']]\
        .sort_values('logmove', ascending=False)
    low_profit_week_mean_price = data_frame[(data_frame['week'] > 120)\
        & (data_frame['week'] < 140)].price.mean()
    other_week_mean_price = data_frame[(data_frame['week'] < 120)\
        | (data_frame['week'] > 140)].price.mean()
    return [highest_sales, data_frame.describe(), low_profit_week_mean_price, other_week_mean_price]

def models(data_frame):
    '''
    Runs models
    Returns models and model summaries
    '''
    df1 = data_frame.copy()

    label_binarizer = LabelBinarizer()

    label_binarizer_output = label_binarizer.fit_transform(df1['brand'])

    dummy_brand = pd.DataFrame(label_binarizer_output,
                            columns = label_binarizer.classes_)

    model_df = data_frame.join(dummy_brand)
    model_df = model_df.drop('brand', axis=1)
    train, test = train_test_split(model_df, test_size=0.2)
    temp = train.select_dtypes(include=[np.number])
    temp = temp[temp.columns.difference(['logmove', 'product'])]
    formula = 'logmove ~ ' + "+".join(temp.columns)
    model = smf.ols(formula, data=train)
    ols_fit = model.fit()

    y_pred = ols_fit.predict(test)
    mean_squared_error(test[['logmove']], y_pred)

    covariate = list(temp.columns.difference(['logmove', 'price']))

    dml_data_bonus = DoubleMLData(train,
                                    y_col='logmove',
                                    d_cols='price',
                                    x_cols=covariate)

    learner = RandomForestRegressor(n_estimators = 100, max_features = 'sqrt', max_depth= 5)

    ml_l_bonus = clone(learner)

    ml_m_bonus = clone(learner)

    learner = LassoCV()

    np.random.seed(3141)

    obj_dml_plr = DoubleMLPLR(dml_data_bonus, ml_l_bonus, ml_m_bonus)

    obj_dml_plr.fit()

    return [ols_fit, obj_dml_plr]
