# ML-tools-explorations

In this project, I explored three extensions for three different machine learning algorithms respectively.

 - Extension 1: Logistic regression with L1(Lasso) regularization
 - Extension 2: Support Vector Machine with Gaussian Kernel
 - Extension 3:  Neural Network with SoftMax as the outer layer activation function.

For each extension, I chose two different datasets; one dataset was used in class, and another one is a new dataset also available in the Scikit-learn library. For each dataset, I implemented two Scikit-defined (1. With extension 2. Without extension) models and one self-defined model.(With extension.)

Therefore, I have two notebooks under each extension task:
1.	Scikit-learn
2.	extended self-defined implementation.

A total of 6 accuracy scores will be shown, 3 scores for each dataset.

# Extension 1: Lasso on Logistic Regression

We learned ridge regression and lasso regression in the class and implemented ridge regression in the homework. Those are linear regression with different regularization terms added to reduce overfitting problem.

L1 regularization: Lasso. L2 regularization: Ridge.

In this extension, I explored the logistic regression model in Scikit-learn library. By setting the penalty to None, we emulate the logistic regression which developed in the homework. Then we apply the penalty term l1 (Lasso) to experiment logistic regression with l1 regularization term. Lastly, I implemented it from scratch with homework codes.

**Detail explanations and implementations of Lasso can be found in both notebooks under Extension 1 Folder.**

## Table of Accuracy:
For both datasets, the Scikit-learn logisticRegression model achieved higher accuracy when applying l1 penalty term than not applying any penalty term.

My implementation for both datasets was also achieved high accuracy as well, which indicates that adding lasso regularization can improve our model.

## Analysis
The reason why the accuracy increases while applying lasso regularization:

1. Reduce the overfitting problem and generalize the model.
2. Lasso shrinks the less important features’ coefficients to zero thus, removing some features altogether. So it’s ideal for feature selection.
3. After applying lasso, previous misclassified data were correctly labeled because of the reduction of noisy features.

# Extension 2: Gaussian Kernel on SVM

Gaussian Kernel is very popular and heavily used in Support Vector Machine Algorithm. I experimented the Scikit-learn SVC model to run linear kernel and Gaussian Kernel separately. Then I implement Gaussian Kernel with homework codes. Instead of using linear kernel, I modified the Kernel function to Gaussian and changed kernel_svm to utilize gaussian kernel instead. Finally I made a change in the prediction function: f_dual.

**Detail explanations and implementations of Gaussian Kernel can be found in both notebooks under Extension 2 Folder.**

## Table of Accuracy:
Accuracy increased on both datasets if we changed to Gaussian Kernel
0.97 increases to 0.98 for breast cancer data, 0.95 increases to 0.97 for fetch_20newsgroups.
My implementation for both datasets was also achieved high accuracy score.

## Analysis
For both datasets, we set Scikit-learn SVC to linear Kernel to emulate what we implement in our homework, then switch to rbf as our extensions. Even though it improved slightly on accuracy, the gaussian kernel is still very powerful. Gaussian Kernel transforms our feature space to a new Z space that is linearly separable, and our SVM finds the best decision boundary that has the largest margin to the support vectors, which increases the accuracy rate for prediction.

**Notice:**
we don’t compute the Z-space; Instead, we utilize Kernel trick to achieve the same effects with less computation!
For dataset fetch_20newsgroups, each sample contains 130107 features! If we want to apply polynomial feature transformation, that’s way more expensive in terms of computation. And clearly, the dataset isn’t linearly separable. We need a transformation to make our dataset linearly separable, and Gaussian Kernel is the key point. Because of it, we can classify such large feature size datasets very effectively.

# Extension 3: SoftMax on Neural Network

SoftMax is a function that takes input vector of K real numbers and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.
**Detail explanations and implementations of SoftMax can be found in both notebooks under Extension 3 Folder.**


Instead of using Scikit-learn library Neural Network Model: MLPClassifier, I used Keras to build a simple neural network for two reasons.

1. I can learn more powerful library not only Scikit-learn library
2. Keras is easy to build a flexible NN model where I can change the structure of my neural network. (add layers, change neurons numbers, configure parameters). MLPClassifier in Scikit-learn instead doesn’t provide such flexibility. (not support SoftMax for activation, no control of input/output layer.)

## Table of Accuracy:

Accuracy improved by SoftMax: 0.9759 > 0.872 on Load_digits dataset. However, for the fetch_covtype dataset, NN with SoftMax didn’t perform well as comparing to NN with Relu.
My implementation with SoftMax performed pretty well on load_digit dataset, much better than what I implemented in the NN homework, which can only achieve accuracy of 0.2.

The reason for low accuracy for Fetch_covtype dataset is under ***Analysis*** section.

## Analysis
For dataset load_digits, Both NN Relu and NN SoftMax are under the same configurations. Accuracy increases from 0.872 to 0.9759, which indicates that SoftMax is good at deciding final outputs. Since this dataset is a multi-class problem, which also shows that SoftMax is really powerful when dealing with such problems.
For dataset Fetch_covtype, the same configurations applied to NN Relu and NN SoftMax.
Accuracy decreases from 0.82 to 0.771. One reason could be that SoftMax isn’t suitable for this dataset. Moreover, I only did 10 epochs for each Neural Network, which might be insufficient for SoftMax.
**My Implementing of SoftMax:**
For dataset load_digits, my model run 1000 iterations for training, which takes 2min 17s
However, for dataset Fetch_covtype, my model only ran 100 iterations because of time. 1h 12 mins to run 100 iterations. It took so much time to iterate over all training examples for each iteration. This dataset contains 406708 samples for X_train. Because of under-training, the accuracy can only achieve 0.55. We need to improve the algorithm to train quickly and increase the number of iterations so that the model can perform much better.
