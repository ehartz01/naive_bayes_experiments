#hw 2 for machine learning
#ethan hartzell
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from math import log, exp
import random

#this is the naive bayes classifier
class n_bayes:
	#it takes its training set in the constructor
	def __init__(self, data): 	#takes an array of strings
		self.vocab = set()
		self.word_counts = defaultdict(float) #word:count
		self.class_counts = defaultdict(float)	#class:count
		self.word_given_class = defaultdict(float) ##(word,class):count
		self.word_given_class_prob = defaultdict(float) #(word,class):logprob
		self.corpus = [] #training set
		self.class_given_word = defaultdict(float) #(class,word):logprob
		self.class_prob = defaultdict(float) #class : logprob
		self.smoothing = 1.0
		self.word_class_counts = defaultdict(float)
		for line in data:	#separate sentence from label
			l = line[:-1].split()
			c = line[-1]
			for word in l:
				self.vocab.add(word)	#keep set of vocab
				self.word_counts[word] += 1	#increment word coutn
			self.corpus.append((l,c))

		for line, label in self.corpus:
			for word in line:
				self.word_given_class[(word,label)] += 1	#count word given label
				self.word_class_counts[label] += 1
			self.class_counts[label] += 1					#count class
	#this calculates log probabilities for words and classes
	def train(self, laplace_val = 1):
		self.smoothing = laplace_val
		if laplace_val != 0:
			self.word_given_class_prob = defaultdict(lambda:(log(laplace_val)-log(laplace_val*len(self.vocab))))
		for word,label in self.word_given_class.keys():
			self.word_given_class_prob[(word,label)] = log((self.word_given_class[(word,label)] + laplace_val)/(self.word_class_counts[label] + laplace_val*len(self.vocab)))
		total = len(self.corpus)
		for label in self.class_counts.keys():
			self.class_prob[label] = log(self.word_class_counts[label]/total)

	#this takes a sentence and compares the probabilities for each class and picks the highest one
	def classify(self, example): 	#takes a sentence as input
		possible_labels = []
		label_vals = []
		label_dict = {} #val : label

		for label in self.class_counts.keys():	#for each possible label
			possible_labels.append(label)		#put the label in an array
			wordprob = 0
			for word in example.split():				#for each word in the example for each class add the prob of the word given the class
				if word in self.vocab:
					wordprob += self.word_given_class_prob[(word,label)]
							#elif self.smoothing != 0:
				#		wordprob += log(self.smoothing) - log(self.smoothing*len(self.vocab))
			label_vals.append(self.class_prob[label] + wordprob)	#put the log prob of that label + logprob of words|label in a corresponding array index
			
		for count, val in enumerate(label_vals):
			label_dict[val] = possible_labels[count]

		return label_dict[max(label_dict.keys())]
	#this runs classify on a large set of examples
	def classify_data_set(self, data): # takes an array of unlabeled sentences
		labeled_data = []
		for line in data:
			labeled_data.append((line, self.classify(line)))
		return labeled_data

#this will return training and test sets organized from crossfold validation and an unlabeled version of the test set
def get_train_and_test(file,num_of_folds=10):
	f = open(file)
	tmp = f.readlines()
	pos_ex = []
	neg_ex = []
	#separate and shuffle positive and negative examples
	for line in tmp:
		newline = ""
		for word in line:
			if word != "0" and word != "1":
				newline += word
			if word == "0":
				newline += word
				neg_ex.append(newline)
			if word == "1":
				newline += word
				pos_ex.append(newline)
	folds = generate_folds(pos_ex,neg_ex,num_of_folds)

	data_sets = tenfold_val(folds,num_of_folds)
	return data_sets

#merge the examples
def generate_folds(pos_ex, neg_ex,num_of_folds):
	random.shuffle(pos_ex)
	random.shuffle(neg_ex)

	posfolds = []
	negfolds = []
	folds = []
	posx = np.asarray(pos_ex)
	negx = np.asarray(neg_ex)
	for arr in np.array_split(posx,10):
		posfolds.append(list(arr))
	for arr in np.array_split(negx,10):
		negfolds.append(list(arr))

	for count, i in enumerate(posfolds):
		fold = []
		for j in i:
			fold.append(j)
		for j in negfolds[count]:
			fold.append(j)
		random.shuffle(fold)
		folds.append(fold)
	random.shuffle(folds)
	return folds

#you generate lists of training and test sets which should be of differing sizes
def tenfold_val(folds,num_of_folds):
	trainings = [] #holds arrays of training sets
	tests = [] #holds arrays of matching test sets at same indices
	unlabeled_tests = [] #same as tests but without the labels at the end

	for i in range(0,num_of_folds):
		tests.append(folds[i])
		newtest = []
		for count, fold in enumerate(folds):
			if count != i:
				newtest += fold
		trainings.append(newtest)

	for tset in tests:
		newarr = []
		for sent in tset:
			newsent = sent[:-1]
			newarr.append(newsent)
		unlabeled_tests.append(newarr)

	return (trainings, tests, unlabeled_tests)

#compares data to the version of the data with the correct labels and gives an accuracy rate
def evaluate(classified_data, original_labeled):
	correct = 0.0
	total = 0.0
	for count, (sent, label) in enumerate(classified_data):
		if label == original_labeled[count][-1]:
			correct += 1
		total += 1
	return correct/total

#evaluates many sets
def get_accuracy(trainings, tests, unlabeled_tests,laplace_val=1):
	correct_percentages = []
	for count, training_set in enumerate(trainings):
		classifier = n_bayes(training_set)
		classifier.train(laplace_val)
		new_labeled_data = classifier.classify_data_set(unlabeled_tests[count])
		correct_percentages.append(evaluate(new_labeled_data,tests[count]))
	return correct_percentages

#evaluates sets for smaller and larger sized data used in experiment 1
def get_accuracy_bysize(trainings, tests, unlabeled_tests,laplace_val=1):
	correct_percentages = []
	for count, training_set in enumerate(trainings):
		accuracies_by_size = [] #starting from smallest
		for i in range(1,11,1):
			newsized_trainset = ( training_set[:i*len(training_set)/(10)] )
			classifier = n_bayes(newsized_trainset)
			classifier.train(laplace_val)
			new_labeled_data = classifier.classify_data_set(unlabeled_tests[count])
			accuracies_by_size.append(evaluate(new_labeled_data,tests[count]))
		correct_percentages.append(accuracies_by_size)
	return correct_percentages	#we return a list of trials by .1-1, we need to to average all the .1s together, and all the .2s etc

#returns the mean and standard deviation of a set of accuracy rates
#takes a list of percentages as input
def get_mean_and_sd(percentages):
	total = 0.0
	for percentage in percentages:
		total += percentage
	mean = total/len(percentages)
	sd_total = 0.0
	for percentage in percentages:
		sd_total += (percentage - mean)**2
	sd = sd_total/len(percentages)
	sd = sd**0.5
	return (mean,sd)

#this is experiment one
#it will return means and standard deviations for different trials
def experiment(dset, laplace_val):

	percents = get_accuracy_bysize(dset[0],dset[1],dset[2],laplace_val)
	reorganized = []
	for i in range(0,10):
		newarray = []
		for plist in percents:
			newarray.append(plist[i])
		reorganized.append(newarray)
	stats_by_size = [] #will hold (mean, sd) organized by size (.1, .2 ... 1)
	for i in reorganized:
		stats_by_size.append(get_mean_and_sd(i))
	return stats_by_size

f1 = "yelp_labelled.txt"
f2 = "imdb_labelled.txt"
f3 = "amazon_cells_labelled.txt"

#run the experiment on yelp data for two different smoothing values
dset = get_train_and_test(f1)
thing_to_plot_0 = experiment(dset,0)
thing_to_plot_1 = experiment(dset,1)
y1 = []
y1err = []
for mean,sd in thing_to_plot_0:
	y1.append(mean)
	y1err.append(sd)
y2 = []
y2err = []
for mean,sd in thing_to_plot_1:
	y2.append(mean)
	y2err.append(sd)
x = []
for i in range(90,990,90):
	x.append(i)
plt.figure()
plt.errorbar(x,y1,y1err)
plt.errorbar(x,y2,y2err)
plt.title("Naive bayes on Yelp (blue m=0, green m=1)")
plt.xlabel("training set size")
plt.ylabel("accuracy")
plt.show()

#do it again on imdb data
dset = get_train_and_test(f2)
thing_to_plot_0 = experiment(dset,0)
thing_to_plot_1 = experiment(dset,1)
y1 = []
y1err = []
for mean,sd in thing_to_plot_0:
	y1.append(mean)
	y1err.append(sd)
y2 = []
y2err = []
for mean,sd in thing_to_plot_1:
	y2.append(mean)
	y2err.append(sd)
x = []
for i in range(90,990,90):
	x.append(i)
plt.figure()
plt.errorbar(x,y1,y1err)
plt.errorbar(x,y2,y2err)
plt.title("Naive bayes on imdb (blue m=0, green m=1)")
plt.xlabel("training set size")
plt.ylabel("accuracy")
plt.show()

#do it on amazon data
dset = get_train_and_test(f3)
thing_to_plot_0 = experiment(dset,0)
thing_to_plot_1 = experiment(dset,1)
y1 = []
y1err = []
for mean,sd in thing_to_plot_0:
	y1.append(mean)
	y1err.append(sd)
y2 = []
y2err = []
for mean,sd in thing_to_plot_1:
	y2.append(mean)
	y2err.append(sd)
x = []
for i in range(90,990,90):
	x.append(i)
plt.figure()
plt.errorbar(x,y1,y1err)
plt.errorbar(x,y2,y2err)
plt.title("Naive bayes on amazon (blue m=0, green m=1)")
plt.xlabel("training set size")
plt.ylabel("accuracy")
plt.show()

#now lets move on to experiment 2
#we will cross validation with different smoothing parameters without any difference in training size
def experiment2(dset, laplace_val):
	meanandsd = get_mean_and_sd( get_accuracy(dset[0],dset[1],dset[2],laplace_val) )
	return meanandsd

#use yelp data to plot
dset = get_train_and_test(f1)
x = []
y1 = []
y1err = []
x = list(np.arange(0,1.1,0.1))

for m in x:
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
for m in range(2,11,1):
	x.append(m)
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
plt.figure()
plt.errorbar(x,y1,y1err)
plt.title("yelp data averages w/ varied smoothing")
plt.xlabel("m (laplace smoothing value)")
plt.ylabel("average accuarcy")
plt.show()

#do it again with imdb data
dset = get_train_and_test(f2)
x = []
y1 = []
y1err = []
x = list(np.arange(0,1.1,0.1))

for m in x:
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
for m in range(2,11,1):
	x.append(m)
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
plt.figure()
plt.errorbar(x,y1,y1err)
plt.title("imdb data averages w/ varied smoothing")
plt.xlabel("m (laplace smoothing value)")
plt.ylabel("average accuarcy")
plt.show()

#do it again with amazon data
dset = get_train_and_test(f3)
x = []
y1 = []
y1err = []
x = list(np.arange(0,1.1,0.1))

for m in x:
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
for m in range(2,11,1):
	x.append(m)
	tmp = experiment2(dset,m)
	y1.append(tmp[0])
	y1err.append(tmp[1])
plt.figure()
plt.errorbar(x,y1,y1err)
plt.title("amazon data averages w/ varied smoothing")
plt.xlabel("m (laplace smoothing value)")
plt.ylabel("average accuarcy")
plt.show()