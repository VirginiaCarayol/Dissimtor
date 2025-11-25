# -*- coding: utf-8 -*-

import argparse

parser=argparse.ArgumentParser(prog='Training_ann_dissimtor',description='Create an artificial neural network (ANN) for specific HLA-I allele.')
parser.add_argument(type=str, help="HLA-I allele for which you want to create an artificial neural network (ANN).", dest='allele')
parser.add_argument("--show-hyperparameter-skf", help="For the best hyperparameters chosen by the author, show the stratified k-fold (SKF) cross-validation.",
                    action="store_true", dest="show_hyperparameter_skf")
parser.add_argument("--show-plots", help="Show plots of curves of loss vs. epoch when --show-hyperparameter-skf is used.",
                    action="store_true", dest="show_plots")
parser.add_argument("--save-test-subset", help="Save the test subset (which has not been used to create and train the saved model).", 
                    action="store_true", dest="save_test_subset")

args=parser.parse_args()

allele=args.allele
show_plots=args.show_plots
show_hyperparameter_skf=args.show_hyperparameter_skf
save_test_subset=args.save_test_subset

print("Importing modules...")
from matplotlib import pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys
import warnings

warnings.filterwarnings('ignore')

# Set this environment variable to silence these warnings from tensorflow:
#
#   oneDNN custom operations are on. You may see slightly different numerical 
#   results due to floating-point round-off errors from different computation
#   orders. To turn them off, set the environment variable
#   `TF_ENABLE_ONEDNN_OPTS=0`. 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import tensorflow as tf

print("Modules imported.")

# Construct the name of the file expected to contain the data related to the 
# specified allele ('binders').
file_allele=os.path.join("datasets", f"{allele}_iedb_netmhc_binder.txt")

hla_I=pd.read_csv(file_allele, sep=" ")

# Similar to the previous step, another file called 'random_nmers.txt' is read
# ('non binders').
random=pd.read_csv(os.path.join("datasets", "random_nmers.txt"))

# Prepare training datasets: amino acid transformation using BLOSUM50.
A="5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0,"
R="-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3,"
N="-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3,"
D="-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4,"
C="-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1,"
Q="-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3,"
E="-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3,"
G="0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4,"
H="-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4,"
I="-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4,"
L="-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1,"
K="-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3,"
M="-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1,"
F="-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1,"
P="-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3,"
S="1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2,"
T="0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0,"
W="-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3,"
Y="-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1,"
V="0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5,"
X="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"

## Binders: 
hla_Idf=pd.DataFrame(hla_I, columns=[allele])

for (character, aminoacid) in [("A", A),
                               ("R", R),
                               ("N", N),
                               ("D", D),
                               ("C", C),
                               ("Q", Q),
                               ("E", E),
                               ("G", G),
                               ("H", H),
                               ("I", I),
                               ("L", L),
                               ("K", K),
                               ("M", M),
                               ("F", F),
                               ("P", P),
                               ("S", S),
                               ("T", T),
                               ("W", W),
                               ("Y", Y),
                               ("V", V),
                               ("X", X)]:
   hla_Idf[allele]=hla_Idf[allele].str.replace(character, aminoacid)

# Remake the dataframe so each amino acid has it's values split into separate
# columns.
hla_Idf=hla_Idf[allele].str.split(',', expand=True)

# Add a column, called 'Binder' with value 1 for all rows (all these peptides 
# bind to that HLA-I allele).
hla_Idf['Binder']=1

# Calculate the number of rows in hla_Idf and multiply it by 10 to determine how
# many samples select from the 'random' dataframe.
count_HLArandom=hla_Idf.shape[0] * 10

## Non binders:
randomdf=pd.DataFrame(random, columns=['Random'])

# Shuffle the rows of 'randomdf'.
randomdf=randomdf.sample(frac=1, random_state=42)

# Randomly select 'n' samples from randomdf, where 'n' is 10 times the number of 
# rows in hla_Idf:
randomdf=randomdf.sample(n=count_HLArandom)

# Amino acid transformation using BLOSUM50.
for (character, aminoacid) in [("A", A),
                               ("R", R),
                               ("N", N),
                               ("D", D),
                               ("C", C),
                               ("Q", Q),
                               ("E", E),
                               ("G", G),
                               ("H", H),
                               ("I", I),
                               ("L", L),
                               ("K", K),
                               ("M", M),
                               ("F", F),
                               ("P", P),
                               ("S", S),
                               ("T", T),
                               ("W", W),
                               ("Y", Y),
                               ("V", V),
                               ("X", X)]:
  randomdf['Random']=randomdf['Random'].str.replace(character, aminoacid)

# Remake the dataframe so each amino acid has it's values split into separate
# columns.
randomdf=randomdf['Random'].str.split(',', expand=True)

# Add a column, called 'Binder' with value 0 for all rows (all these peptides do not
# bind to that HLA-I allele).
randomdf['Binder']=0

## Choosing the best hyperparameters for the machine learning model.

# The previous dataframes are concatenated, forming the training set 'train_df'.
train_df=pd.concat([hla_Idf,randomdf])

# Some modifications of the dataset:
# Remove the last column because it's always empty. This is because the amino acids
# have trailing commas.
train_df=train_df.drop(columns=[180])  
# Rename the columns by prefixing them with "P".
train_df.columns=[f"P{str(i)}" for i in train_df.columns]  
train_df=train_df.apply(pd.to_numeric)  
train_df=train_df.sample(frac=1, random_state=42)  

# Split the dataset into three subsets: 'train', 'validation' and 'test'.

# The training set data is used to train the model. A model is usually trained using 
# an iterative process. At each iteration, a performance measure is calculated 
# that reflects the error made by the model when applied to the training set data. 
# This measure is used to update the model parameters in order to reduce the error of
# the model when applied to the data in the training set. Model parameters are a set
# of variables associated with the model and their values ​​are learned during the 
# training process. In addition to model parameters, there are other variables 
# associated with a model whose values ​​are not updated during training. These variables
# are called hyperparameters. The optimal or near-optimal values ​​of the hyperparameters
# are determined using the validation set data. This process is often called hyperparameter 
# tuning. After the model is trained and tuned, the test set data is used to evaluate 
# the model's ability to generalize (i.e., performance on unseen data). 

# A cross-validation method has been used. In particular, the stratified k-fold 
# cross validation (SKF) has been used:

# 1) The dataset is split into a training dataset and a test dataset.

# 2) The training dataset is split into K-folds.

# 3) Within the K-folds, (K-1) fold is used for training.

# 4) 1 fold is used for validation.

# 5) The model with specific hyperparameters is trained on training data (K-1 folds) 
# and evaluated on validation data (1 fold). The performance of the model is saved.

# 6) The above steps (step 3, step 4, and step 5) are repeated until each of the k-folds
# is used for validation purposes. 

# 7) Finally, the mean and standard deviation of the model performance on the validation
# set are calculated by taking all the model scores calculated in step 5 for each of
# the K models.

# 8) Steps 3 to 7 are repeated for different hyperparameter values.

# 9) Finally, the hyperparameters that result in the most optimal validation set mean 
# and standard deviation of the model scores are selected.

# 10) The model is then trained using the entire training dataset (step 2; the data
# used as training and validation) and the model performance is calculated on the
# test dataset (step 1).

## STEP 1. 
# train_test_split() function is used, which splits the 'train_df' dataset into a 
# training-validation and test set in a random and stratified manner (maintaining
# the class proportions in both datasets). 

train_validation_df1, test_df1=train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['PBinder'],
    random_state=42
)

## STEPS 2-7.

# Parameters.
rd=42
partitions=10 

# Perform the stratified partition with the StratifiedKFold() function.
skf=StratifiedKFold(n_splits=partitions, shuffle=True, random_state=rd)

# Save the input (features, variables or characteristics) and output (class or label) 
# in two variables, called 'x_skf' and 'y_skf', respectively:
x_skf, y_skf=train_validation_df1.iloc[:,0:180], train_validation_df1['PBinder']

for train, validation in skf.split(x_skf, y_skf):
  x_train_skf, x_validation_skf, y_train_skf, y_validation_skf=x_skf.iloc[train], x_skf.iloc[validation], y_skf.iloc[train], y_skf.iloc[validation]

# In each partition or 'fold', the model is trained with the training subset (K-1 folds) 
# and evaluated on the corresponding validation subset (1 fold).

def create_model(learning_rate):
    """Function to create an artificial neural network using tensorflow."""
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu', name='Hidden1'))
    model.add(tf.keras.layers.Dense(10, activation='relu', name='Hidden2'))
    model.add(tf.keras.layers.Dense(1, name='Output'))
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model

def train_model(model, x, y, epochs, batch_size):
  """Function to train the model."""
  history=model.fit(x=x,
                    y=y,
                    batch_size=batch_size,
                    epochs=epochs)
  # The list of epochs is stored separately from the rest of history.
  epochs=history.epoch
  return epochs, history

def plot_the_loss_curve(epochs, mae_training, mae_validation):
  """Function to plot a curve of loss vs. epoch for both the training and validation subsets."""
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean squared error")
  plt.plot(epochs[1:], mae_training[1:], label="Training loss")
  plt.plot(epochs[1:], mae_validation[1:], label="Validation loss")
  plt.legend()
  merged_mae_lists=mae_training[1:] + mae_validation[1:]
  highest_loss=max(merged_mae_lists)
  lowest_loss=min(merged_mae_lists)
  delta=highest_loss - lowest_loss
  print(delta)
  top_of_y_axis=highest_loss + (delta * 0.05)
  bottom_of_y_axis=lowest_loss - (delta * 0.05)
  plt.ylim([bottom_of_y_axis, top_of_y_axis])
  plt.show()

# Best hyperparameters chosen by the author:
LR=0.01  # Learning rate
BS=1000  # Batch size
EPOCHS=50 

# Store the MSE of each subpartition (from both the training and validation subset), 
# calculate the mean and standard deviation of the MSE and display these results.

if show_hyperparameter_skf:
  print("Doing stratified k-fold cross-validation with the best hyperparameters chosen by the author...")
  # List to store the results of each partition.
  mse_val=[]
  mse_train=[]

  # Stratified k-fold cross-validation loop:
  for train_index, val_index in skf.split(x_skf, y_skf):
      x_train, x_val=x_skf.iloc[train_index], x_skf.iloc[val_index]
      y_train, y_val=y_skf.iloc[train_index], y_skf.iloc[val_index]

      # Create and train the model with the hyperparameters defined above.
      model=create_model(LR)
      history=model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BS, shuffle=True,
                          validation_data=(x_val, y_val))

      training_mse=history.history['mean_squared_error']
      validation_mse=history.history['val_mean_squared_error']

      if show_plots:
        plot_the_loss_curve(history.epoch, training_mse, validation_mse)

      # Get the MSE on the training and validation set of the current partition.
      train_mse=training_mse[-1]
      val_mse=validation_mse[-1]

      # Save the results (MSE on training and validation set) of all partitions.
      mse_train.append(train_mse)
      mse_val.append(val_mse)

  # Calculate the mean and standard deviation of the MSE of the training and validation 
  # set in the K models:
  mean_mse_train=np.mean(mse_train)
  std_mse_train=np.std(mse_train)

  mean_mse_val=np.mean(mse_val)
  std_mse_val=np.std(mse_val)

  print("Results of the stratified k-fold cross-validation with the best hyperparameters chosen by the author:")

  print(f"Training MSE mean: {mean_mse_train:.4f}")
  print(f"Training MSE standard deviation: {std_mse_train:.4f}")

  print(f"Validation MSE mean: {mean_mse_val:.4f}")
  print(f"Validation MSE standard deviation: {std_mse_val:.4f}")

## STEPS 8 and 9. 

# Different hyperparameter values: the hyperparameters that result in the most optimal 
# validation set mean and standard deviation of the model scores are selected (shown as 
# 'best hyperparameters chosen by the author').

# Learning rate. If the learning rate is very small, it takes many calculations to reach 
# the value that produces the minimum loss. However, if it is too large, we can exceed 
# the value that produces the minimum loss and even reach a value that produces a larger
# loss than before.

# Batch size. The batch size is a gradient descent hyperparameter that controls how 
# many training samples to work with before the internal parameters of the model are 
# updated.

# Epochs. The number of epochs is a gradient descent hyperparameter that controls the 
# number of complete passes through the training dataset.

# Number of hidden layers and neurons (modifying the function that creates the model).

## STEP 10. 

print("Creating and training the model (artificial neural network (ANN))...")

# Build and train the model with those chosen hyperparameters using the entire training set 
# (training + validation; 'train_validation_df1') and evaluate it on the test set ('test_df1').

# Training set: save the input ('features', variables or characteristics) and output (class, 
# label or 'label') in two variables, called 'x_skf' and 'y_skf', respectively:
x_skf, y_skf=train_validation_df1.iloc[:,0:180], train_validation_df1['PBinder']

model=create_model(LR)

epochs, history=train_model(model=model, x=x_skf, y=y_skf, epochs=EPOCHS, batch_size=BS)

print("Evaluating the model in the test set...")

# Test set: save the input ('features', variables or characteristics) and output (class, label
# or 'label') in two variables, called 'x_test_skf' and 'y_test_skf', respectively:
x_test_skf, y_test_skf=test_df1.iloc[:,0:180], test_df1['PBinder']

test_loss, test_mse=model.evaluate(x_test_skf, y_test_skf, batch_size=BS)

print(f"Test loss: {test_loss}")
print(f"Test MSE: {test_mse}")

# Save test subset:
if save_test_subset:
  test_filename = os.path.join("ANN_alleles", "test_subset", f"test_df1_{allele}.csv")
  test_df1.to_csv(test_filename, index=False)

# Get the predictions for the test set:
y_pred=model.predict(x_test_skf)

# Convert the predicted scores to binary labels: we have to choose a threshold
# from which the classifier is forced to make the prediction as a "positive class" 
# (in this case, binder) whenever its confidence is greater than said threshold. 
# 'threshold = 0.75'
y_pred_binary=np.where(y_pred > 0.75, 1, 0)

# Create a dataframe with the real and predictions labels.
df_pred=pd.DataFrame({'y_pred': np.array(y_pred).flatten(), 'label_test': np.array(y_test_skf).flatten()})

# Confusion matrix:
confusion_SFK=tf.math.confusion_matrix(y_test_skf, y_pred_binary, num_classes=2)

TN_SKF=confusion_SFK[0,0]  # True negatives
FP_SFK=confusion_SFK[0,1]  # False positives
FN_SFK=confusion_SFK[1,0]  # False negatives
TP_SFK=confusion_SFK[1,1]  # True positives

# False positive rate (FPR) and false negative rate (FNR).
FPR_SFK=FP_SFK / (FP_SFK + TN_SKF)
FNR_SFK=FN_SFK / (FN_SFK + TP_SFK)

print(confusion_SFK)
print(f"TN_SFK: {TN_SKF}, FP_SFK: {FP_SFK}, FN_SFK: {FN_SFK}, TP_SFK: {TP_SFK}")
print(f"FPR_SFK: {FPR_SFK}, FNR_SFK: {FNR_SFK}")

# Other metrics: accuracy, recall o sensitivity, specificity, precision y F1.
print(sklearn.metrics.classification_report(y_test_skf, y_pred_binary, output_dict=True))

# ROC curve and AUC. 
# Calculate the scores representing the affinity of a peptide for a specific HLA-I allele.
y_pred_prob=model.predict(x_test_skf).flatten()

# Get the ROC curve values. roc_curve() function of scikit-learn get the true positive 
# and false positive rates at all thresholds.
fpr, tpr, thresholds=roc_curve(y_test_skf, y_pred_prob)

# Calculate the AUC:
auc_value=auc(fpr, tpr)
print(f"AUC: {auc_value}")

print("Saving the results...")

# Save the final model.
model_file=os.path.join("ANN_alleles", f"dissimtor_{allele}.keras")

print(f"All done. Saving model in {model_file}")

model.save(model_file)

print("Completed.")