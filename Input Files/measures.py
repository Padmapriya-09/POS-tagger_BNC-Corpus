import os
import pandas as pd

d=os.path.abspath('..')

fn = input("Enter the name of the confusion matrix file\n(File should be in Output Files folder): ")
os.chdir(d+"/Output Files")
try:
	f=open(fn,'r',encoding='utf8')
except OSError:
	print("Could not open/read file")
	print("First test the model by using the command- \"python testing.py\"")
	sys.exit()


confusion_matrix = pd.read_csv(f,index_col=0,header=0)
f.close()
print('Completed reading confusion matrix')

no_of_times_predicted=[]
F1_score=[]
print("Tag"+'\t\t'+"precision"+'\t\t'+"recall"+'\t\t'+"F-score")
for tag in confusion_matrix.index.values.tolist():
	TP=confusion_matrix.at[tag,tag]
	row_list=confusion_matrix.loc[tag,:].values.tolist()
	FN=sum(row_list)-TP
	column_list=confusion_matrix.loc[:,tag].values.tolist()
	FP=sum(column_list)-TP
	if TP+FP==0:
		p=0
	else:
		p=TP/(TP+FP)
	if TP+FN==0:
		r=0
	else:
		r=TP/(TP+FN)
	if p+r==0:
		Fscore=0
	else:
		Fscore = 2 * (p * r) / (p+r)
	no_of_times_predicted.append(sum(column_list))
	F1_score.append(Fscore)
	print(tag+'\t\t'+str(p)+'\t\t'+str(r)+'\t\t'+str(Fscore))

print("Macro F1 score: "+str(sum(F1_score)/len(F1_score)))
num=0
den=0
for (n,f) in zip(no_of_times_predicted,F1_score):
	num+=(n*f)
	den+=n
print("Weighted F1 score: "+str(num/den))