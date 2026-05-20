# run_cv_all_models.py

# ================================
# IMPORTS
# ================================
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Import models
from models import (
    get_dnn_model,
    get_cnn1d_model,
    get_bilstm_model,
    get_gru_model,
    get_bigru_model
)

# ================================
# DEVICE SELECTION (for logging)
# ================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ================================
# PATHS
# ================================
RIC_PATH = "/Users/apple/Documents/Academics/001-SZIIT-Research/IR_Code_Data/3_Features_tr_te/features_train/Insulin_PSSM_RICLBP_2476.csv"
QLC_PATH = "/Users/apple/Documents/Academics/001-SZIIT-Research/IR_Code_Data/3_Features_tr_te/features_train/Insulin_QLC_2476.csv"
ESM2_PATH = "/Users/apple/Documents/Academics/001-SZIIT-Research/IR_Code_Data/3_Features_tr_te/features_train/ESM2_embeddings_2476.csv"
ProtT5_PATH = "/Users/apple/Documents/Academics/001-SZIIT-Research/IR_Code_Data/3_Features_tr_te/features_train/ProtT5_embeddings_2476.csv"

RESULT_DIR = "/Users/apple/Documents/Academics/001-SZIIT-Research/PaperData/Insulin_60PercentCDHIT/RESULTS/"

FeatureName = 'Optimal_Full'

# ================================
# LOAD DATA
# ================================

df1 = pd.read_csv(RIC_PATH, header=None)
df2 = pd.read_csv(QLC_PATH, header=None)
df3 = pd.read_csv(ESM2_PATH, header=None)
df4 = pd.read_csv(ProtT5_PATH, header=None)

RIC = df1.iloc[:, :-1]
QLC = df2.iloc[:, :-1]
ESM2 = df3.iloc[:, :-1]
ProtT5 = df4.iloc[:, :-1]

X = pd.concat([RIC, QLC, ESM2], axis=1)
y = df1.iloc[:, -1]

print("Features:", X.shape)
print("Labels:", y.shape)

# ================================
# PREPROCESSING (UNCHANGED)
# ================================
scaler = StandardScaler()

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())
X = scaler.fit_transform(X)

def prepare_input(model_name, X_scaled):
    """
    Returns:
    - X_model: reshaped input
    - config: input_shape for the model
    """

    if model_name == "DNN":
        X_train = X_scaled
        config = (X_scaled.shape[1],)

    elif model_name == "CNN1D":
        vector_len = X_scaled.shape[1]
        X_train = X_scaled.reshape((X_scaled.shape[0], vector_len, 1))
        config = (vector_len, 1)

    elif model_name in ["BiLSTM", "GRU", "BiGRU"]:
        X_train = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        config = (1, X_scaled.shape[1])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return X_train, config
# ================================
# MODEL REGISTRY
# ================================
MODEL_REGISTRY = {
    "DNN": get_dnn_model,
    "CNN1D": get_cnn1d_model,
    "BiLSTM": get_bilstm_model,
    "GRU": get_gru_model,
    "BiGRU": get_bigru_model
}

# ================================
# CV FUNCTION
# ================================
def run_10fold_cv(model_name, model_fn, X_train, y_train, config):

    skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)

    acc, f1, prec, rec = [], [], [], []
    spec, mcc, auc = [], [], []
    all_y_true, all_y_proba = [], []


    for train_index, val_index in skfold.split(X_train, y_train):
        # Split the data into training and validation sets for the current fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model = model_fn(config)
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

        model.fit(X_train_fold, y_train_fold, epochs=40, batch_size=16, verbose=0,validation_split=0.1, callbacks=reduce_lr)
    
        y_val_pred = model.predict(X_val_fold)
        
        all_y_true.extend(y_val_fold)
        all_y_proba.extend(y_val_pred)
        
        y_val_pred_binary = (y_val_pred > 0.64).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred_binary).ravel()
        specificity = tn / (tn + fp)

        acc.append(accuracy_score(y_val_fold, y_val_pred_binary))
        prec.append(precision_score(y_val_fold, y_val_pred_binary))
        rec.append(recall_score(y_val_fold, y_val_pred_binary))
        spec.append(specificity)
        f1.append(f1_score(y_val_fold, y_val_pred_binary))
        mcc.append(matthews_corrcoef(y_val_fold, y_val_pred_binary))
        auc.append(roc_auc_score(y_val_fold, y_val_pred))


    # ================================
    # SAVE PREDICTIONS
    # ================================
    pred_df = pd.DataFrame({
        "true_label": all_y_true,
        "pred_prob": all_y_proba
    })

    pred_path = f"{RESULT_DIR}{model_name}_{FeatureName}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print(f"  Saved predictions → {pred_path}")
    
    # ================================
    # PRINT MEAN PERFORMANCE (AFTER 10 FOLDS)
    # ================================
    print(f"\n=== Mean 10-Fold Performance: {model_name} ===")
    print(f"Accuracy = {np.mean(acc)}")
    print(f"F1-score = {np.mean(f1)}")
    print(f"Precision = {np.mean(prec)}")
    print(f"Recall (SN) = {np.mean(rec)}")
    print(f"Specificity = {np.mean(spec)}")
    print(f"MCC = {np.mean(mcc)}")
    print(f"AUC = {np.mean(auc)}")

    # ================================
    # RETURN MEAN METRICS
    # ================================
    return {
        "Model": model_name,
        "Accuracy": np.mean(acc),
        "F1": np.mean(f1),
        "Precision": np.mean(prec),
        "Recall": np.mean(rec),
        "Specificity": np.mean(spec),
        "MCC": np.mean(mcc),
        "AUC": np.mean(auc)
    }

# ================================
# RUN ALL MODELS
# ================================
all_results = []

for model_name, model_fn in MODEL_REGISTRY.items():
    print(f"\nRunning 10-Fold CV for {model_name}")
    X_model, config = prepare_input(model_name, X)
    result = run_10fold_cv(model_name, model_fn, X_model, y, config)
    all_results.append(result)

# ================================
# SAVE FINAL RESULTS
# ================================
results_df = pd.DataFrame(all_results)
final_csv = f"{RESULT_DIR}ALL_MODELS_{FeatureName}_10FOLD_RESULTS.csv"
results_df.to_csv(final_csv, index=False)

print("\n=== FINAL SUMMARY ===")
print(results_df)
print(f"\nSaved summary → {final_csv}")