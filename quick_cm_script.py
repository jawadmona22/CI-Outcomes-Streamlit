import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
file_path = "C:\\Users\\jawad\\Downloads\\Clinical Comparison - ML and Clinicians (1).xlsx"

ml_df = pd.read_excel(file_path,sheet_name=0)
print(ml_df)
clin_df = pd.read_excel(file_path,sheet_name=1)

comp_df = pd.DataFrame()

patients = ["P8","P14","P17","P18","P19","P20","P22",'P23','P24','P26','P27','P28','P30','P31','P32']
true_list = []
pred_list = []
for patient in patients:
    ID = int(patient.split('P')[1])

    true_azbio = ml_df[ml_df["Participant Number"] == ID]
    Y = int(true_azbio["AzBio Quiet "].iloc[0])

    for col in clin_df.columns:
        if patient in col and "AzBio Quiet" in col:
            preds = clin_df[col].dropna()

            for p in preds:
                pred_list.append(int(p))
                true_list.append(Y)

print(len(true_list), len(pred_list))
print(true_list[:10])
print(pred_list[:10])

bins = [0, 20, 40, 60,80,101]
labels = ["0-20","21-40","41-60","61-80","81-100"]

true_binned = pd.cut(true_list, bins=bins, labels=labels, include_lowest=True)
pred_binned = pd.cut(pred_list, bins=bins, labels=labels, include_lowest=True)


mask = ~(true_binned.isna() | pred_binned.isna())
true_binned = true_binned[mask]
pred_binned = pred_binned[mask]
cm = confusion_matrix(true_binned, pred_binned)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Reds")
plt.title("Binned AzBio Quiet Confusion Matrix")
plt.show()


ml_preds = ml_df["Pred AzBio Quiet"]
true = ml_df["AzBio Quiet "]

true_binned = pd.cut(true, bins=bins, labels=labels, include_lowest=True)
pred_binned = pd.cut(ml_preds, bins=bins, labels=labels, include_lowest=True)


mask = ~(true_binned.isna() | pred_binned.isna())
true_binned = true_binned[mask]
pred_binned = pred_binned[mask]
cm = confusion_matrix(true_binned, pred_binned)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Reds")
plt.title("ML AzBio Quiet Confusion Matrix")
plt.show()
