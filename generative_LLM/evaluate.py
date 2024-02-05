
import torch
import codecs
from torchmetrics import Accuracy,F1Score,Precision, Recall
from sklearn.metrics import accuracy_score, balanced_accuracy_score,average_precision_score,recall_score,classification_report,precision_score


def evaluate(y_true, y_pred, exp_dir,message):
    preds_test = torch.tensor(y_pred)  # converting predictions and labels into torch Tensors  instead of pandas Series to pass them to torch metrics
    labels_test = torch.tensor(y_true)

    with codecs.open(exp_dir + "results.txt", 'w', encoding='utf8') as out:
        out.write(message+"\n\n")
        print("sklearn's accuracy score = " + str(accuracy_score(y_true, y_pred, normalize=True)))
        out.write("\nsklearn's accuracy score = " + str(accuracy_score(y_true, y_pred, normalize=True)))
        print("sklearn's balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred)))
        out.write("\nsklearn's balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred)))
        print("sklearn's adjusted balanced accuracy score = " + str(balanced_accuracy_score(y_true, y_pred, adjusted=True)))
        out.write("\nsklearn's adjusted balanced accuracy score = " + str(
        balanced_accuracy_score(y_true, y_pred, adjusted=True)))
        print("sklearn's macro precision score = " + str(average_precision_score(y_true, y_pred, average='macro')))
        out.write("\nsklearn's macro precision score = " + str(average_precision_score(y_true, y_pred, average='macro')))
        print("sklearn's macro recall score = " + str(recall_score(y_true, y_pred, average='macro')))
        out.write("\nsklearn's macro recall score = " + str(recall_score(y_true, y_pred, average='macro')))
        print('\033[96m')
        out.write("\nClassification Report:----------------------------\n")
        print(classification_report(y_true, y_pred, labels=[0, 1]))
        out.write(classification_report(y_true, y_pred, labels=[0, 1]))
        print('\033[0m')
        num_classes = 2
        accuracy = Accuracy(average='macro', num_classes=num_classes,task = "binary")
        print("Torch metrics' Accuracy = " + str(accuracy(preds_test, labels_test)))
        out.write("\nTorch metrics' Accuracy = " + str(accuracy(preds_test, labels_test)))
        f1_score = F1Score(average='macro', num_classes=num_classes,task = "binary")
        print("Torch metrics' F1 = " + str(f1_score(preds_test, labels_test)))
        out.write("\nTorch metrics' F1 = " + str(f1_score(preds_test, labels_test)))
        precision = Precision(average='macro', num_classes=num_classes,task = "binary")
        print("Torch metrics' Precision = " + str(precision(preds_test, labels_test)))
        out.write("\nTorch metrics' Precision = " + str(precision(preds_test, labels_test)))
        recall = Recall(average='macro', num_classes=num_classes,task = "binary")
        print("Torch metrics' Recall = " + str(recall(preds_test, labels_test)))
        out.write("\nTorch metrics' Recall = " + str(recall(preds_test, labels_test)))