import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
	def __init__(self,number_groups,number_dimensions=2):
		self.weight = np.zeros((number_groups,number_dimensions+1,1))
		self.alpha = 0.01
		self.number_dimensions=number_dimensions

	def classification(self,values,type_class): #predict
		u = np.matmul(values,self.weight[type_class])
		return 1 if u[0][0]>= 0 else 0

	def training(self,dataframe,type_class): #fit
		for i in range(len(dataframe)):
			input_values=np.array([np.insert(dataframe.iloc[i][:self.number_dimensions].values,0,1)])
			class_value=int(dataframe.iloc[i,self.number_dimensions])
			target = 1 if class_value==type_class else 0
			predict = self.classification(input_values,type_class)
			#weight update
			self.weight[type_class] += self.alpha*(target-predict)*input_values.T
	
	def evaluate(self, dataframe):
		accuracy = []
		for i in range(len(dataframe)):
			input_values=np.array([np.insert(dataframe.iloc[i][:self.number_dimensions].values,0,1)])
			class_value=int(dataframe.iloc[i,self.number_dimensions])
			accuracy.append(self.classification(input_values,class_value))
		return np.mean(accuracy)*100
	
	@staticmethod
	def redefining(value_a,max_a,min_a,max_n,min_n):
		value_n = (value_a - min_a)*(max_n - min_n)/(max_a - min_a) + min_n
		return value_n

	def reorganize(self,df,max_n,min_n):
		for i,column in enumerate(df.columns):
			if i<self.number_dimensions:
				max_a,min_a=np.max(df[column]),np.min(df[column])
				df[column]=self.redefining(df[column],max_a,min_a,max_n,min_n)
		df = df.sample(frac=1).reset_index(drop=True)
		return df

	def plot_data(self,dataframe,animation=False):
		fig, ax = plt.subplots()
		colors=['red','green','blue','yellow','purple']
		classes=dataframe['out_0'].unique()
		for i in classes:
			df_class=dataframe[dataframe['out_0'] == i]
			ax.scatter(df_class['in_0'],df_class['in_1'],color=colors[i])
		for i in classes:
			if self.weight[i,2,0] != 0:
				x = (-1,1)
				y = (-(self.weight[i,0,0] + x[0]*self.weight[i,1,0])/self.weight[i,2,0],
					-(self.weight[i,0,0] + x[1]*self.weight[i,1,0])/self.weight[i,2,0])
				ax.plot(x,y,color=colors[i])
		ax.grid()
		ax.axis('scaled')
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)
		plt.show()


if __name__ == '__main__':
	df=pd.read_csv('data_3groups.csv',sep=';')
	perceptron = Perceptron(number_groups=3,number_dimensions=2)
	df=perceptron.reorganize(df,max_n=1,min_n=-1)

	df_train = df.iloc[:200,:].reset_index(drop=True)
	df_test = df.iloc[300:500,:].reset_index(drop=True)

	for epoch in range(3):
		perceptron.training(dataframe=df_train,type_class=0)
		perceptron.training(dataframe=df_train,type_class=1)
		perceptron.training(dataframe=df_train,type_class=2)

	perceptron.plot_data(df_train)
	print('Accuracy:',perceptron.evaluate(df_test))


