# Code Classifier App

## Description
The aim is to develop an application which will detect the programming language used in a snippet of code using machine learning. It will add to the flavor if we can add some more interpretation, like displaying some confidence score and some key features which lead to the classification.

### Initial Thoughts and Ideas
* All programming languages are written following some structures and rules, so writing some algorithm to detect those rules or heuristics can be an approach, but then writing rules for 30+ languages seems to be a very unfeasible task. Detecting languages based on keywords or file extensions are also ruled out as somebody might pass python code with .txt extension.

* Data credits - We got data thanks to a similar work of @andreasjansson Refer [Link](https://github.com/andreasjansson/language-detection.el/blob/master/snippets.cpkl.gz)

### Experiments Details
Refer [IPython notebook](https://github.com/kr-prince/Code-Classifier-App/blob/main/ProgLang_Classification_ML.ipynb)
* Started with relatively smaller snippets(code within range of len 250-500 chars) to keep the data balanced and do some fast parameter tunings. After that we fed the whole training data for all the languages (around 10000 snippets) to the selected classifier.

* We used TfIdf bi-gram vectors and features capped to 120000. While fitting TfIdf vectors, we clubbed the training data(code snippets) together grouped on languages, instead of keeping all snippets as separate docs.

* Since all symbols and punctuations can be important features in code, so we coded a custom tokenizer which tokenizes words and symbols, but also concatenates repeating symbols, like === or ++

* Results
  
  Classifier Name	| Mean Fit Time(s)	| Mean Test Time(s)	| Mean Train Score	| Mean CV Score	| Best Test Score 
  ----------------|-------------------|-------------------|-------------------|---------------|-----------------
  Logistic Regression	| 44.128	| 0.102	| 0.875	| 0.738	| 0.824
  KNeighbors Classifier	| 0.037	| 21.517	| 0.908	| 0.788	| 0.798
 	SVC Classifier	| 856.900	| 244.200	| 0.585	| 0.481	| 0.814
 	SGD Classifier	| 6.500	| 0.200	| 0.742	| 0.677	| 0.829
 	MultinomialNB Classifier	| 0.400	| 0.100| 	0.892	| 0.801	| 0.810
 	XGB Classifier**	| 1895.000	| 21.600	| 0.938	| 0.785	| 0.797
  
  Due to less Fit and Train time and a reasonably good Test score we selected the good old SGD Classifier
  
  Close analysis of the confusion matrix shows the model working poorly to classify highly closely related programming languages like CPP and C, Java and Csharp. If we remove the import statements and just keep few lines of code it is also non trivial to quickly identify between such close languages. Also snippets like Javascript embedded with HTML or CSS embedded with HTML in the training data are other causes of misclassification. 

## Usage
1. Use conda and install the packages mentioned in *requirements.txt* file, they will include all other required packages.
2. Activate the conda environment, clone the repo and cd into the directory
3. Extract the file `.\data\snippets.zip` to get the pickle form. Run `python CodeClassifierTrain.py` if you want to re-train the classifier locally, else you use the existing model as it is. 
4. Run `python CodeClassifier.py` to test if the classifier and models are working fine.
5. Run `python mainApp.py` to run the python GUI application. 

## App Preview
![Java Code](https://github.com/kr-prince/Code-Classifier-App/blob/main/results/JavaCode.JPG)
![C Code](https://github.com/kr-prince/Code-Classifier-App/blob/main/results/CCode.JPG)
![Prolog Code](https://github.com/kr-prince/Code-Classifier-App/blob/main/results/PrologCode.JPG)

## TODO
* Considering the current results as baseline, explore deep learning approaches
* Better UI/UX and added functionality in the app

Please feel free to add and contribute

