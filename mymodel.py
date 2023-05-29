import uproot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from AdaBoostClassifier import AdaBoostClassifier

def read_root_file(file_path, tree_name):
    root_file = uproot.open(file_path)
    tree = root_file[tree_name]
    vars_to_load_       =   ['MET','METSig','Jet1Pt', 'Jet1Eta', 'Jet1Phi', 'Jet1CSV','Jet2Pt', 'Jet2Eta', 'Jet2Phi', 'Jet2CSV','DiJetMass','DiJetPt', 'DiJetEta','DiJetPhi',"j1j2DR","j1j2Dphi","hMETDphi"]

    df = tree.arrays(vars_to_load_,filter_name="MET>200",library="pd")
    return df


signal_file_paths = glob("../signalFiles/*ggTomonoH_bb_sn_0p35_tn_0p5_mXd_10_MH3_600_*.root")
background_file_paths = glob("../bkfFiles/*ZJetsToNuNu*.root")


signal_dfs = []
for file_path in signal_file_paths:
    df = read_root_file(file_path, "monoHbb_SR_resolved")
    signal_dfs.append(df)

background_dfs = []
for file_path in background_file_paths:
    df = read_root_file(file_path, "monoHbb_SR_resolved")
    background_dfs.append(df)


signal_df = pd.concat(signal_dfs,ignore_index=True)
background_df = pd.concat(background_dfs,ignore_index=True)

# Add a 'target' column to indicate signal (1) or background (0)
signal_df['target'] = 1
background_df['target'] = 0

# Combine the signal and background DataFrames
combined_df = pd.concat([signal_df, background_df])

X = combined_df.drop('target', axis=1)
y = combined_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize and train the custom boosting classifier
boost_classifier = AdaBoostClassifier(n_estimators=50)
boost_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = boost_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate and plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

