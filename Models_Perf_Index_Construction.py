import pandas as pd 
import numpy as np 
import pickle
import glob
import matplotlib.pyplot as plt 

testing_start_date ="2014-12"

def monthly_change(feature):
    return feature.iloc[-1] / feature.iloc[0] - 1
nb_stocksinPort = 30 # Number of stocks we want to include in our portfolio (index)
stock_file= pd.read_csv('Stocks_names.csv')
stocks = list(stock_file['Stocks']) # The list of all the stocks of our univers
models_df = pd.read_csv('Models V2.csv')
models_df=  models_df.set_index('Model')
port_dico ={} # to store the performance metrics of all models
dico_index_cumul_perf ={} # to store all cumul performance of the different models
for c in range(models_df.shape[1]): #models_df.shape[1]
    if c == 0:
        num=11 # because ANN pickles folder has 4 pickles each as the data was divided into 4 parts
    else: 
        num = 21 # because LSTM and GRU  pickles folder have 11 pickles each as the data was divided into 11 parts
    folder = '.\\' + models_df.columns[c] 
    for s in range(1, models_df.shape[0]+1 ): #1, models_df.shape[0]+1   
        folder_list = glob.glob(folder+"\\"+models_df.loc[s,models_df.columns[c]]) # folder of pickles(1 model)
        dico_results_summary = {}
        for i in range(1,num+1):
            with open(folder_list[0]+"\\"+'df_results_summary_'+ models_df.columns[c]+'_strat_'+str(s)+str(i)+'.pickle', 'rb') as handle: 
                temp_df_results_summary = pickle.load(handle)
            dico_results_summary.update(temp_df_results_summary) # This is a dictionnary of where key = stock_name and value = df of prediction and y_test with index monthly dates       
        # Let's build 2 DataFrames 1 for the predictions and 1 for the y_test for corresponding months
        a_random_df = dico_results_summary[stocks[0]] # I just pick any key/value in the dictionary as I am interested in the index (the dates)
        Dates_TestSet = list(a_random_df.index) # list of the dates of the testing set
        perfpredicted ={} #dictionary to stock the predictions
        targets ={} # dictionary to stock the actual returns
        for i in range(len(stocks)): #len(stocks)
            list_pred = []
            list_targets = []
            for j in range(len(Dates_TestSet)):
                temp = dico_results_summary[stocks[i]] # choose value (df) corresponding to the stock name (key)
                list_pred.append(temp.loc[Dates_TestSet[j], 'Pred']) # Add the prediction corresponding to that date
                list_targets.append(temp.loc[Dates_TestSet[j], 'y_test']) # Add the actual perf corresponding to that date
            perfpredicted.update({stocks[i]: list_pred}) # update a column a the df of predictions
            targets.update({stocks[i]: list_targets})   # update a column a the df of y_test
        df_perfpredicted = pd.DataFrame(perfpredicted, index = Dates_TestSet) # df that stores all the predictions per stock for corresponding month
        df_targets = pd.DataFrame(targets, index = Dates_TestSet) # df that stores all the y_test per stock for corresponding month
        
        # ******************************************************Let's build the portfolio now *****************************************************************
        # The objective is to build a portfolio made of the stocks with the best expected return based on our predictions                                      
        # Their weight in the portfolio should be proportional to their expected return (high perf predicted -> high weight, low perf predicted -> low weight)
        #******************************************************************************************************************************************************
        Perfpermonth_predicted = {}   # Dictionay to store the monthly performance of our created portfolio
        for i in range(len(Dates_TestSet)): #len(Dates_TestSet)
            temp_pred = df_perfpredicted.loc[Dates_TestSet[i],:] # store the predicted perf of each asset row by row (month by month)
            temp_target = df_targets.loc[Dates_TestSet[i],:] # store the actual perf of each asset row by row (month by month)
            sort = temp_pred.sort_values(ascending = False) # returns a Series sorted predicted perfomance along with the name of each stock (as index)
            top = sort[:nb_stocksinPort] # select only the top n element to integrate in the portfolio
            exponential_top = np.exp(top) # exponential of the top stocks 
            total = sum(exponential_top) # We sum the value of these elements
            weights = exponential_top / total # We divide each top n elements by that sum (this is the weight of each element)
            idx = exponential_top.index # We obtain the name of these top n elements
            list_temp_target = []
            for j in range(nb_stocksinPort): # number of components
                list_temp_target.append(temp_target.get(idx[j])) #to get the y_test (actual return )value corresponding to the top n elements (best expected performers)
            df_temp_target = pd.DataFrame(list_temp_target, index = idx) # we put the actual return with the stock name as index in a dataframe
            weighted_perf = np.multiply(list_temp_target, weights) # We do an element-wise multiplication of each realized (real) perf by their corresponding weight
            weighted_sum = sum(weighted_perf) # We sum up the weighted perf of each asset corresponding to that date (that's our portfolio perf that month)
            Perfpermonth_predicted.update({Dates_TestSet[i]:weighted_sum}) # We associate that monthly performance to the month in question in a dictionary
        port_returns = (pd.DataFrame(Perfpermonth_predicted.items(), columns=['Dates', 'Port Monthly Perf'])).set_index('Dates') # Here is all the monthly performance of our portfolio reunited in a nice df
        # Let's compute the cumulative performance of the built portfolio (assuming an initial investment of 100)
        cumul_perf = [100]
        for i in range(len(port_returns)):
            perf = cumul_perf[i]*(1+port_returns.iloc[i,0])
            cumul_perf.append(perf)
        temp_cumul_perf = {models_df.columns[c]+'_'+str(s):cumul_perf}
        dico_index_cumul_perf.update(temp_cumul_perf)  # store all cumulative returns corresponding to each model 
        
        port_totalperf = cumul_perf[-1]/cumul_perf[0]-1 # the total performance corresponding of the model
        port_annualized_return = (1+port_totalperf)**(12/60)-1  # Annualizing the return
        # Let's compute some metrics 
        #Standard Deviation
        annualized_volatility =  np.std(port_returns['Port Monthly Perf'])*np.sqrt(12)
        # Parametric Value at Risk
        VaR95 = -1.645*annualized_volatility
        #Sharpe Ratio
        sharpe_ratio = (np.mean(port_returns['Port Monthly Perf']) / np.std(port_returns['Port Monthly Perf']))*np.sqrt(12) # Sharpe ratio
        #MDD
        port_returns["total_return"] = port_returns['Port Monthly Perf'].cumsum()
        port_returns["drawdown"] = port_returns["total_return"] - port_returns["total_return"].cummax()
        MDD = port_returns["drawdown"].min() #Maximum Drawdown
        perf_metrics = [port_annualized_return,annualized_volatility,VaR95,sharpe_ratio,MDD]
        temp_port_dico = {models_df.columns[c]+'_'+str(s):perf_metrics}
        port_dico.update(temp_port_dico)

# This df regroups the performance of all the models by metrics
df_port_metrics =pd.DataFrame(port_dico, index =['Annualized Return','Annualized Volatility','Value at Risk 95','Sharpe Ratio','Max Drawdown'] )

# df_port_metrics.to_csv('.\\Port_Performance_metrics.csv')

# df_port_metrics.to_csv(r'C:\Users\Jonathan\Desktop\ThèseMS\V2\Perf_of_models_by_metrics.csv')
# Now let's determine the best and the worst models according to the metrics computed 
Max_metrics_models= df_port_metrics.idxmax(axis=1) # the models that return the maximum for each row 
Max_metrics_values= df_port_metrics
Min_metrics_models = df_port_metrics.idxmin(axis=1) # the models that return the minimum for each row
list_of_best_models = [Max_metrics_models['Annualized Return'],
                    Min_metrics_models['Annualized Volatility'],
                    Max_metrics_models['Value at Risk 95'], 
                    Max_metrics_models['Sharpe Ratio'],
                    Max_metrics_models['Max Drawdown']]

list_of_worst_models =[Min_metrics_models['Annualized Return'],
                    Max_metrics_models['Annualized Volatility'],
                    Min_metrics_models['Value at Risk 95'], 
                    Min_metrics_models['Sharpe Ratio'],
                    Min_metrics_models['Max Drawdown']]
Best_models_by_metrics = pd.Series(list_of_best_models,index =['Annualized Return','Annualized Volatility','Value at Risk 95','Sharpe Ratio','Max Drawdown'])
Worst_models_by_metrics = pd.Series(list_of_worst_models,index =['Annualized Return','Annualized Volatility','Value at Risk 95','Sharpe Ratio','Max Drawdown'])


# Let's prepare the benchmarks
bench_dico = {} # to store the performance metrics of all the benchmarks 
bench_cumul_dico = {} # to store the cumulative performance of all the benchmarks 
bench_list = pd.read_csv('Benchmarks_names.csv')
bench_dico_returns_list = {}  # to store the returns of each benchmark
for j in range(len(bench_list)):
    benchmark = pd.read_csv(bench_list.iloc[j,0]+'.csv')
    benchmark['Dates'] = pd.to_datetime(benchmark['Dates'], format='%d/%m/%Y').dt.strftime("%Y-%m-%d") #change the date format 
    benchmark['month'] = benchmark['Dates'].astype(str).str[:7] # truncate the month for the right format 
    benchmark.index = benchmark['month'] # use it as an index
    benchmark = benchmark.drop(['month', 'Dates'], axis=1) # drop the unimportant columns
    bench_grouped = benchmark.groupby("month") # group by month
    bench_df_returns = bench_grouped['Price'].apply(monthly_change) # monthly returns
    idx = bench_df_returns.index # get the index 
    index_loc = idx.get_loc(testing_start_date) # get index location of 'testing_start_date'
    bench_df_returns = bench_df_returns.iloc[index_loc+1:] # we add one because we actually starts one month after the 'testing_start_date' set (as we did with predictions)

    bench_cumulperf = [100] # The list where we will store the cumulative performance of the benchmark
    for i in range(len(bench_df_returns)):
        perf = bench_cumulperf[i]*(1+bench_df_returns.iloc[i]) # computing the cumulative performance
        bench_cumulperf.append(perf)
    bench_totalperf = bench_cumulperf[-1]/bench_cumulperf[0]-1
    bench_dico_returns_list.update({bench_list.iloc[j,0]: bench_totalperf})
    bench_annualized_return = (1+bench_totalperf)**(12/60)-1
    #Standard Deviation
    bench_annualized_volatility =  np.std(bench_df_returns)*np.sqrt(12)
    # Value at Risk
    bench_VaR95 = -1.645*bench_annualized_volatility
    #Sharpe Ratio
    bench_sharpe_ratio = (np.mean(bench_df_returns) / np.std(bench_df_returns)) * np.sqrt(12) # Sharpe ratio
    #MDD
    bench_df_returns["total_return"] = bench_df_returns.cumsum()
    bench_df_returns["drawdown"] = bench_df_returns["total_return"] - bench_df_returns["total_return"].cummax()
    bench_MDD = bench_df_returns["drawdown"].min() #Maximum Drawdown
    #####
    bench_perf_metrics = [bench_annualized_return,bench_annualized_volatility,bench_VaR95,bench_sharpe_ratio,bench_MDD]
    temp_bench_dico = {bench_list.iloc[j,0]:bench_perf_metrics} 
    bench_dico.update(temp_bench_dico) # Here we save all the perf metrics by benchmark
    temp_bench_cumul_dico= {bench_list.iloc[j,0]:bench_cumulperf} # Here we save all the cumulative performance by benchmark
    bench_cumul_dico.update(temp_bench_cumul_dico)

df_bench_metrics =pd.DataFrame(bench_dico, index =['Annualized Return','Annualized Volatility','Value at Risk 95','Sharpe Ratio','Max Drawdown'] )


listofmonths =list(idx[index_loc:]) # to use as index for our dataframe of cumulative perf
df_bench_cumul_perf = pd.DataFrame(bench_cumul_dico, index=listofmonths)

# This df regroups the cumul return of all the models tried 
df_port_cumul_perf = pd.DataFrame(dico_index_cumul_perf, index= listofmonths) 
df_port_cumul_perf['Average perf of all models'] = df_port_cumul_perf.apply(np.mean, axis=1)
#let's get the returns of each month for the aggregated portfolio
aggregate_ret_list=[]
for i in range(df_port_cumul_perf.shape[0]-1):
    ret = df_port_cumul_perf.iloc[i+1,-1]/df_port_cumul_perf.iloc[i,-1]-1
    aggregate_ret_list.append(ret)

# let's compute the metrics of df_port_cumul_perf['Average perf of all models']
aggregateval_totalperf = df_port_cumul_perf.iloc[-1,-1]/df_port_cumul_perf.iloc[0,-1]-1 # total per
aggregate_port_annualized_return = (1+aggregateval_totalperf)**(12/60)-1  # Annualizing the return

# Let's compute some metrics 
#Standard Deviation
aggregate_annualized_volatility =  np.std(aggregate_ret_list)*np.sqrt(12)
# Parametric Value at Risk
aggregate_VaR95 = -1.645*aggregate_annualized_volatility
#Sharpe Ratio
aggregate_sharpe_ratio = (np.mean(aggregate_ret_list) / np.std(aggregate_ret_list))*np.sqrt(12) # Sharpe ratio
#MDD
aggregate_df = pd.DataFrame(aggregate_ret_list, index = Dates_TestSet, columns = ['ret'])
aggregate_df["total_return"] = aggregate_df['ret'].cumsum()
aggregate_df["drawdown"] = aggregate_df["total_return"] - aggregate_df["total_return"].cummax()
aggregate_MDD = aggregate_df["drawdown"].min() #Maximum Drawdown

summary_aggregate = [aggregate_port_annualized_return,aggregate_annualized_volatility, aggregate_VaR95,aggregate_sharpe_ratio,aggregate_MDD]
df_bench_metrics['Aggregated AI index'] = summary_aggregate
# df_bench_metrics.to_csv('.\\Benchmarks_Perf_by_metrics.csv')

dico_returns_list ={} # to store the total cumul performance of each model
for i in range(df_port_cumul_perf.shape[1]):
    dico_returns_list.update({df_port_cumul_perf.columns[i]:df_port_cumul_perf.iloc[-1,i]/df_port_cumul_perf.iloc[0,i]-1 })

dico_returns_bench_plus_aggregate= {}
for i in range(df_bench_cumul_perf.shape[1]):
    dico_returns_bench_plus_aggregate.update({df_bench_cumul_perf.columns[i]:df_bench_cumul_perf.iloc[-1,i]/df_bench_cumul_perf.iloc[0,i]-1 })

# plt.figure(figsize=(10,6))
# # plt.plot(df_port_cumul_perf['ANN_1'], color='red', label='ANN(64)')
# # plt.plot(df_port_cumul_perf['LSTM_1'], color='blue', label='LSTM(64)')
# plt.plot(df_port_cumul_perf['Average perf of all models'], color='red', label='AI Aggregated Index')
# plt.plot(df_bench_cumul_perf['Proprietary Index 1'],color='blue', label='Proprietary Index 1')
# plt.plot(df_bench_cumul_perf['Proprietary Index 2'],color='green', label='Proprietary Index 2')
# plt.plot(df_bench_cumul_perf['Proprietary Index 3'],color='orange',label ='Proprietary Index 3')
# plt.plot(df_bench_cumul_perf['SPY'],color='grey',label ='S&P500')
# plt.xticks(rotation=90)
# plt.tick_params(axis='x',which='major',labelsize=7.5)
# plt.title('AI Aggregated Index vs Benchmarks', fontweight='bold')  #'AI Based Index vs Proprietary Indices & US Market'
# plt.ylabel('Performance')
# plt.legend()
# plt.show()