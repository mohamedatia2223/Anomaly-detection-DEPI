import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.svm import OneClassSVM
from tensorflow.keras import layers, regularizers, models, callbacks
import warnings
warnings.filterwarnings('ignore')
import joblib

try:
    import keras_tuner as kt
except Exception:
    import kerastuner as kt


columns = [
    "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl","dttl",
    "sloss","dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb",
    "smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt",
    "Dintpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
    "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
    "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","label"
]

files = [
    r"C:\games\git\Anomaly-detection-DEPI\src\data\UNSW-NB15\raw\UNSW-NB15_1.csv",
    r"C:\games\git\Anomaly-detection-DEPI\src\data\UNSW-NB15\raw\UNSW-NB15_2.csv",
    r"C:\games\git\Anomaly-detection-DEPI\src\data\UNSW-NB15\raw\UNSW-NB15_3.csv",
    r"C:\games\git\Anomaly-detection-DEPI\src\data\UNSW-NB15\raw\UNSW-NB15_4.csv"
]

dfs = [pd.read_csv(f, header=None, names=columns) for f in files]

df_full = pd.concat(dfs, ignore_index=True)

print("Loading data...")
df = df_full
y = df['label'].values

features_to_encode = ['proto', 'state', 'service']
df_encoded = pd.get_dummies(df, columns=features_to_encode, prefix=features_to_encode)
df = df_encoded.drop(columns=['label'], axis=1)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].copy()

if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

X_train_unsup = X[y == 0].copy() if y is not None else X.copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_unsup)
X_all_scaled = scaler.transform(X.values)

input_dim = X_train_scaled.shape[1]
print("Input dim:", input_dim)

np.save("labels.npy", y)
joblib.dump(scaler, "scaler.pkl")

np.save("feature_columns.npy", np.array(num_cols))

# -------------------------
# Build model 
# -------------------------
def build_autoencoder_sequential(hp):
    units1 = hp.Int('units1', min_value=64, max_value=512, step=64)
    units2 = hp.Int('units2', min_value=32, max_value=256, step=32)
    latent = hp.Int('latent', min_value=8, max_value=max(8, input_dim // 4), step=max(1, (input_dim // 4)//4))
    l1_strength = hp.Float('l1', 1e-6, 1e-3, sampling='log')
    lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')

    model = keras.Sequential()

    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.GaussianNoise(0.1))

    model.add(layers.Dense(units1, activation=None, activity_regularizer=regularizers.l1(l1_strength)))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Dense(units2, activation=None, activity_regularizer=regularizers.l1(l1_strength)))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Dense(latent, activation=None, activity_regularizer=regularizers.l1(l1_strength), name='bottleneck'))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Dense(units2, activation=None))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Dense(units1, activation=None))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Dense(input_dim, activation='linear'))

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# -------------------------
# Tuner 
# -------------------------
tuner = kt.RandomSearch(
    build_autoencoder_sequential,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='kt_dir_seq',
    project_name='ae_unsw_seq',
    overwrite=True
)

print("Starting hyperparameter search (Sequential model)...")
es_to_tuner = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

tuner.search(
    X_train_scaled, X_train_scaled,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es_to_tuner],
    verbose=2
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(f" units1: {best_hps.get('units1')}")
print(f" units2: {best_hps.get('units2')}")
print(f" latent: {best_hps.get('latent')}")
print(f" l1: {best_hps.get('l1'):.2e}")
print(f" lr: {best_hps.get('lr'):.2e}")


# -------------------------
# Build & train final best model
# -------------------------
best_model = tuner.hypermodel.build(best_hps)

final_es = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history = best_model.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[final_es],
    verbose=2
)
best_model.save("autoencoder_model.h5")


# -------------------------
# Evaluation
# -------------------------
recon_all = best_model.predict(X_all_scaled, verbose=1)
mse = np.mean(np.square(recon_all - X_all_scaled), axis=1)

train_recon = best_model.predict(X_train_scaled, verbose=1)
train_mse = np.mean(np.square(train_recon - X_train_scaled), axis=1)

threshold = np.percentile(train_mse, 95)
print(f"Anomaly threshold (95th percentile): {threshold:.4f}")

anomaly_flag = (mse > threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y, anomaly_flag))
print("\nClassification Report:")
print(classification_report(y, anomaly_flag))

plt.figure(figsize=(8,5))
plt.hist(mse[y == 0], bins=50, alpha=0.7, label='Normal')
plt.hist(mse[y == 1], bins=50, alpha=0.7, label='Attack')
plt.axvline(threshold, color='k', linestyle='--', label=f"thr={threshold:.4f}")
plt.legend(); plt.show()

print("\nConfusion Matrix:")
print(confusion_matrix(y, anomaly_flag))

plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y, anomaly_flag), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()