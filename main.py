# This is try at making a machine learning model, supervised
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from sklearn.model_selection import train_test_split


def CLT_prob(data, prob_value):
    """
    This is to return the prob of a value less 
    prob_value
    
    Arguments: 
    data -- this is supposed to be a dataframe of values
    prob_value -- the probability of a value
    
    RETURNS:
    percentage rounded to two decimals places"""

    #mean
    mean = data.mean()
    
    #standard deviation
    standard_deviation = data.std()

    #z
    z = (prob_value - mean) / standard_deviation
    probability = st.norm.cdf(z)

    return round(probability * 100, 2)

def data_map(data):
    plt.plot(data.iloc[:,-2],data.iloc[:,-1])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    plt.close()

def check_corr(corr):
    """Checks the features with high correlation to find variables
    that are not independant
    
    Parameters:
    corr -- a DataFrame of the correlation values to see check independant variables
    however 0 covariance does not necessarily mean they are independant, the opposite is true.
    if they are independant then covariance must be zero
    
    Returns:
    high_corr_list -- a list of integers which map to indexes of columns of dataframe with high correlation"""
    corr_significance = 0.8
    lenght = corr.shape[1]
    
    high_corr_list = []
    for i in range(0, lenght - 1):
        for j in range(i + 1,lenght):
            if ((-corr_significance >= corr.values[i][j]) or (corr.values[i][j] >= corr_significance)) and (j not in high_corr_list):
                high_corr_list.extend([j])
    return high_corr_list

def naive_bayes_theorem(data, new_instance):
    """The classifying feature must be the last column"""
    
    # finding correlations
    corr = data.iloc[:,:-1].corr(method="pearson")
    
    #datamap
    # sns.heatmap(corr,vmax=1,vmin=-.5,square=True,linewidths=.2)
    # data_map(data)
    
    #checking correlation and dropping dependant variables
    index_drop_list = check_corr(corr)
    data.drop(data.iloc[:,index_drop_list],axis=1)
    new_instance.drop(new_instance.iloc[:,index_drop_list],axis=1)
    
    #calculating probability
    labels = sorted(list(data.iloc[:,-1].unique()))
    prob_list = []
    for i in labels:
        prob_list.append(calc_class_prob(data,new_instance,i))
    
    #choosing class from probability
    lenght = new_instance.shape[0]
    chosen_labels = [1]*lenght
    for i in range(lenght):
        prob = 0
        for j in range(len(prob_list)):
            if prob_list[j][i] > prob:
                prob = prob_list[j][i]
                chosen_labels[i] = labels[j]
    
    return chosen_labels

# probability calculations
def calc_class_prob(data, new_instance,id_class):
    """Calculates the probability of a class
    
    Parameters:
    data -- DataFrame of the data
    new_instance -- instance we're trying to classify
    id_class -- the class 
    
    Returns: the probability of the class"""
    prior = calc_prior(data,id_class)
    prob_features_given_y = calc_gaussian_prob(data, new_instance, id_class)

    # multiplying by prior
    prob_y_given_features = [i * prior for i in prob_features_given_y]
    

    return prob_y_given_features

    
def calc_gaussian_prob(df,new_instance,id_class):
    """Calculates the p(x1,x2...|id_class) of all instances assuming normal gaussian distribution
    
    Parameters:
    df -- the dataframe
    new_instance -- the new instance dataframe
    id_class -- the selected class

    Return: 
    return_list -- a list float number corresponding to the p_allx_of_y of the features given the class
    of all instances, the list corresponds to p_allx_of_y of each instance based on 1 class"""
    #dropping instances that don't match class
    df = df.drop(df[ df[ df.columns[-1] ] != id_class ].index)

    #calculating std and mean to make it efficient
    columns = df.shape[1] - 1
    std, mean = [1]*(columns), [1]*(columns)
    for i in range(0,columns):
        std[i] = df.iloc[:,i].std()
        mean[i] = df.iloc[:,i].mean()

    # - 1 because without the classifying feature
    return_list = []
    for j in range(new_instance.shape[0]):
        # naive bayes calculation
        p_xi_given_y = []
        for i in range(0,columns):
            x = new_instance.iloc[j,i]
            prob = (1 / (np.sqrt(2 * np.pi) * std[i])) * np.exp( (-1/2) * (((x - mean[i]) / std[i]) )** 2)
            p_xi_given_y.append(prob)
        
        p_allx_given_y = 1
        for i in p_xi_given_y:
            p_allx_given_y *= i
        return_list.append(p_allx_given_y)

    return return_list
    

def calc_prior(data,y):
    """Returns probability of class y"""
    instances_lenght = len(data)
    
    y_count = list(data["diagnosis"]).count(y)
    p_y = y_count / instances_lenght
    return p_y


###########
def main():
    new_instance = pd.DataFrame(data=pd.read_csv("new_instance.csv"))
    data = pd.read_csv("Breast_cancer_data.csv")
    new_instance.columns = data.columns
    train, test = train_test_split(data, test_size=.2, random_state=41)

    X_test = test.iloc[:,:-1].values
    Y_test = test.iloc[:,-1].values
    Y_pred = naive_bayes_theorem(train, test)

    from sklearn.metrics import confusion_matrix, f1_score
    print(confusion_matrix(Y_test, Y_pred))
    print(f1_score(Y_test, Y_pred))
    return 0

if __name__ == '__main__':
    main()