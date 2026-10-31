
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



dataset = pd.read_csv(r"C:\Users\ybali\Downloads\logit classification.csv")

x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,  test_size = 0.20, random_state = 0)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', C = 1.0, solver = 'liblinear')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)




train_accuracy = classifier.score(x_train, y_train)
test_accuracy = classifier.score(x_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)



from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

bias = classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test, y_test)
print(variance)





##dataset1 = pd.read_csv(r"C:\Users\ybali\Downloads\final1.csv")

#d2 = dataset1.copy()

#dataset1 = dataset1.iloc[:,3:5].values

#from sklearn.preprocessing import StandardScaler
#sc =  StandardScaler()
#m= sc.fit_transform(dataset1)

#y_pred1 = pd.DataFrame()


#d2 ['y_pred1']  =  classifier.predict(m)

#d2.to_csv('final1.csv')



from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score = roc_auc_score(y_test,y_pred_prob)
auc_score

fpr,tpr, thresholds = roc_curve(y_test,y_pred_prob)


import os

from pathlib import Path

project_path = Path(__file__).resolve().parent
dataset_folder = project_path / "dataset"/"images"
dataset_folder.mkdir(parents=True, exist_ok=True)

image_path = dataset_folder / "roc_curve.png"


os.makedirs("dataset/images", exist_ok=True)






plt.figure(figsize = (8,6))
plt.plot(fpr,tpr, label = f'Logistic Regression (AUC =  {auc_score:.2f})')
plt.plot([0,1], [0,1],  'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True positive Rate')
plt.title('ROC Curve')
plt.legend(loc = 'lower right')
plt.grid()














    
plt.savefig(
        
        "dataset/images/roc_curve.png",
        dpi = 300,
        bbox_inches= 'tight'
        
        
    )
plt.show()
print("ROC curve saved successfully in the 'images' folder.")



    