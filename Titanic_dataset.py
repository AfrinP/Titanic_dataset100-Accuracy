import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load Training and Test Data
train_raw_data = pd.read_csv('train.csv')
test_raw_data = pd.read_csv('test.csv')
y_raw_test = pd.read_csv('gender_submission.csv')

# Function for preprocessing
def preprocessing(data):
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    data.loc[data['Age'].isnull(), ['Age']] = data['Age'].median()
    data.loc[data['Fare'].isnull(), ['Fare']] = data['Age'].mean()
    data = pd.get_dummies(data) 
    
    try:
        data.pop('Survived')
    except:
        pass
    
    
    X_columns = data.columns
    normalize = StandardScaler()
    X = normalize.fit_transform(data)
    X = pd.DataFrame(X, columns = X_columns)
    return X 

# Train Model
def train(train_raw_data):
    X_train = preprocessing(train_raw_data)
    y_train = train_raw_data["Survived"]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

# Initilaize Training and Test Data
X_train = preprocessing(train_raw_data)
y_train = train_raw_data['Survived']
X_test = preprocessing(test_raw_data)
y_test = pd.merge(test_raw_data, y_raw_test, on = 'PassengerId')['Survived']


# Pipeline to choose best among many hyperparameters
pipeline = Pipeline([('clf', LogisticRegression())])
parameters = {
        'clf__C': (0.01, 0.001,0.1,1.0,5.0, 10),
        'clf__max_iter' : (100, 1000, 10000),
        'clf__class_weight' : ('balanced', None),
        'clf__penalty' : ('l1', 'l2'),
    }
grid_search = GridSearchCV(pipeline, parameters, n_jobs= -1, verbose = 1, scoring = 'accuracy');
grid_search.fit(X_train, y_train);

# Print Accuracy on Test set
print("Accuracy on test set is", str(grid_search.score(X_test, y_test)*100) + '%')

