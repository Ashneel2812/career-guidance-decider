import joblib#for loading and saving trained models
import pandas as pd#for creating and handling dataframes
import seaborn as sns#for advanced plots
import matplotlib.pyplot as plt#for plotting

                #DATA COLLECTION
data = pd.read_csv('synthetic_data.csv')

# print("shape of the given dataset is : ",data.shape)#gives shape of dataset
# print("size of the given dataset is : ",data.size)#gives size of dataset
# print("checking for NaN values in the given dataset : ",data.columns[data.isna().any()])
# print(data['mbti_type'].unique())

                #FEATURE SELECTION
x = data.iloc[:,:-1].values#choosing input columns
y = data.iloc[:,-1].values#choosing output column

                #FEATURE ENGINEERING
from sklearn.model_selection import train_test_split
#splitting dataset into train and test partitions for evaluating model after training
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#scaling the data
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
joblib.dump(sc, '1_scaler.pkl')#saving the scaler into hard disk

                #MODEL TRAINING
from sklearn.svm import SVC#importing algorithm
svm_model = SVC(probability=True)#initialising the algorithm
svm_model.fit(x_train_scaled,y_train)#training the algorithm on dataset
joblib.dump(svm_model, '1_svm_model.pkl')#saving the trained model

                #MODEL EVALUATION
from sklearn.metrics import classification_report
y_pred = svm_model.predict(x_test_scaled)#creating a classification report
print("classification report : \n",classification_report(y_test,y_pred))#displaying the classification report

from sklearn.metrics import confusion_matrix
svm_cf = confusion_matrix(y_test,y_pred)#creating a confusion matrix

classes = ['INTP', 'ENTP', 'ISFP', 'INFJ',
           'ESTJ', 'ENFP', 'ESTP', 'INFP',
           'ISTP', 'ENTJ', 'ESFJ', 'ESFP',
           'ISFJ', 'INTJ', 'ENFJ', 'ISTJ']
sns.heatmap(svm_cf,annot=True,fmt='d',cmap='Blues',
            xticklabels=classes,yticklabels=classes)#plotting the confusion matrix as a heatmap
plt.savefig('svm_confusion_matrix.png',bbox_inches='tight')#saving the confusion matrix
plt.title("SVM CONFUSION MATRIX")
plt.show()#displaying the plot