import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

# load data
df = pd.read_csv('./data/kidney_disease.csv')

print df.info()

# change strings to digits
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'target'},inplace=True)
df['pe'] = df['pe'].replace(to_replace='good',value=0)
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
df.drop('id',axis=1,inplace=True)

print df.head()
# check NaN/null values
print df.isnull().sum()
# data cleaning, drop NaN
df2 = df.dropna(axis=0)
print(df2.shape)
df2.head()
print df2['target'].value_counts()
sns.countplot(x="target", data=df2, palette="bwr")
plt.savefig('imgs/target_values.png')
#plt.show()
# We lost 2/3 of original data because of NaN values
# idea: try imputation, but only the important features

# first correct some incorrect data
df2['wc'] = df2['wc'].replace("\t6200",6200)
df2['wc'] = df2['wc'].replace("\t8400",8400)
# df2[['wc']]=df2.wc.replace("\t6200",6200)
# df2.wc=df2.wc.replace("\t8400",8400)

# check outliers
fig = plt.figure(figsize = (12, 5))
ax1 = fig.add_subplot(1, 5, 1)
sns.boxplot(x= df2["target"], y=df2["age"], data=df2)
ax2 = fig.add_subplot(1, 5, 2)
sns.boxplot(x= df2["target"],  y=df2["bp"], data=df2)
ax3 = fig.add_subplot(1, 5, 3)
sns.boxplot(x= df2["target"],  y=df2["bu"], data=df2)
ax4 = fig.add_subplot(1, 5, 4)
sns.boxplot( x= df2["target"], y=df2["bgr"], data=df2)
ax5 = fig.add_subplot(1, 5, 5)
sns.boxplot( x= df2["target"], y=df2["sod"], data=df2)
plt.savefig('imgs/boxplot.png')
#plt.show()
df2[['wc','pcv','rc']] = df2[['wc','pcv','rc']].astype(float)
print df2.info()
# plt.hist(df2.wc)
#plt.show()
df2 = df2[df2['wc']<=20000]
print df2.info()
# make index series continues
df2.index=range(0,len(df2),1)
print df2.head(5)

#HEAT MAP #correlation of parameters
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df2.corr(),annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="blue")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig('imgs/correlation.png')

# Split the dataset into training and testing set
x_data = df2.drop(['target'], axis = 1)
y = df2['target']
# Normalize
X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle= True, random_state=5)
print X_train.head(5)



# Linear Regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)
print("Linear Regression - Test Accuracy: {:.2f}%".format(linear_regression.score(X_test,y_test)*100))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
print("Logistic Regression - Test Accuracy: {:.2f}%".format(logistic_regression.score(X_test,y_test)*100))

# SVM
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(X_train, y_train)
print("SVM - Test Accuracy: {:.2f}%".format(svm.score(X_test,y_test)*100))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
print("Naive Bayes - Test Accuracy: {:.2f}%".format(naive_bayes.score(X_test,y_test)*100))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
print("Decision Tree - Test Accuracy: {:.2f}%".format(decision_tree.score(X_test, y_test)*100))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)  # n_neighbors means k
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print("KNN (with k = 5) - Test Accuracy: {:.2f}%".format( knn.score(X_test, y_test)*100))

'''
# Neural Network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=5,max_iter=10,verbose=10,learning_rate_init=.1)
#mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=5,max_iter=10,verbose=10,learning_rate_init=.1)
mlp.fit(X_train, y_train)
print("Neural Network - Test Accuracy: {:.2f}%".format( mlp.score(X_test, y_test)*100))
'''

# select top 15 important features
from sklearn.feature_selection import SelectKBest, chi2
feature_select = SelectKBest(chi2, k = 15)
feature_select.fit(X_train, y_train)
score_list = feature_select.scores_
top_features = X_train.columns
uni_features = list(zip(score_list, top_features))
print(sorted(uni_features, reverse=True)[0:15])


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_train,y_train)
print("Random Forest - Test Accuracy: {:.2f}%".format(clf.score(X_test,y_test)*100))

#
features_list = X_train.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(5, 7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()


