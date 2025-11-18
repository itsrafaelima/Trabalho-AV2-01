#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
Implementa pipeline completo para estimativa de consumo energético:
- EDA (estatísticas, histogramas, dispersões, correlações)
- Pré-processamento (StandardScaler)
- Construção e treino de MLP (3 -> 10 -> 1) com Keras (fallback sklearn)
- Early stopping (validação 10% do treino), até 300 épocas
- Avaliação: MSE, MAE, R2
- Geração de gráficos e salvamento de artefatos
Uso:
    python app.py --data caminho/consumo_energia_full.csv
Ou apenas:
    python app.py
que tentará carregar arquivos padrão em ./data/
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime

# ML libs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try tensorflow.keras, fallback to sklearn's MLPRegressor if not available
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception as e:
    USE_TF = False
    from sklearn.neural_network import MLPRegressor

# -----------------------
# Utility / I/O
# -----------------------
OUTDIR = Path("output")
PLOTS_DIR = OUTDIR / "plots"
MODELS_DIR = OUTDIR / "models"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def auto_load_csv(path_candidates):
    """Tenta carregar o primeiro CSV existente na lista."""
    for p in path_candidates:
        if os.path.exists(p):
            print(f"[i] Carregando: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"Nenhum dos arquivos foi encontrado: {path_candidates}")

def detect_and_rename(df):
    """Detecta colunas x1,x2,x3,y por nomes comuns em pt/en e renomeia."""
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    possibles = {
        'x1': ['x1','temperatura','temperatura ambiente','temp','temperature','temp_c','t'],
        'x2': ['x2','umidade','umidade relativa','humidity','hum','h'],
        'x3': ['x3','ocupacao','ocupação','ocupantes','people','occupancy','ocupacao_predio'],
        'y':  ['y','consumo','consumo de energia','energia','consumo_kwh','kwh','energy_consumption']
    }
    for key, opts in possibles.items():
        for o in opts:
            if o in cols:
                mapping[cols[o]] = key
                break
    # if nothing matched and df has 4 columns, assume order
    if len(mapping) < 4 and df.shape[1] == 4:
        mapping = {df.columns[0]:'x1', df.columns[1]:'x2', df.columns[2]:'x3', df.columns[3]:'y'}
    df2 = df.rename(columns=mapping)
    missing = [c for c in ['x1','x2','x3','y'] if c not in df2.columns]
    if missing:
        raise ValueError(f"Não foi possível mapear colunas automaticamente. Colunas faltando: {missing}. Colunas disponíveis: {list(df.columns)}")
    return df2[['x1','x2','x3','y']].copy()

# -----------------------
# EDA
# -----------------------
def do_eda(df):
    """Gera estatísticas e gráficos (salva em output). Retorna descritores e correlação."""
    desc = df.describe().T
    desc.to_csv(OUTDIR / "descriptive_stats.csv")
    # histograms
    for col in df.columns:
        plt.figure()
        plt.hist(df[col].values, bins=30)
        plt.title(f"Histograma de {col}")
        plt.xlabel(col); plt.ylabel("Frequência")
        plt.savefig(PLOTS_DIR / f"hist_{col}.png", bbox_inches='tight')
        plt.close()
    # scatter x vs y
    for col in ['x1','x2','x3']:
        plt.figure()
        plt.scatter(df[col].values, df['y'].values, s=10, alpha=0.6)
        plt.title(f"Dispersão: {col} vs y")
        plt.xlabel(col); plt.ylabel("y (consumo kWh)")
        plt.savefig(PLOTS_DIR / f"scatter_{col}_y.png", bbox_inches='tight')
        plt.close()
    # correlation
    corr = df.corr()
    corr.to_csv(OUTDIR / "correlation_matrix.csv")
    # save a small png with correlation heatmap
    try:
        import seaborn as sns
        plt.figure(figsize=(6,5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlação")
        plt.savefig(PLOTS_DIR / "correlation_matrix.png", bbox_inches='tight')
        plt.close()
    except Exception:
        # fallback: text csv is fine
        pass
    return desc, corr

# -----------------------
# Preprocessing
# -----------------------
def preprocess(df, test_size=0.2, random_state=42):
    """Aplica StandardScaler (X e y), faz split 80/20 e retorna scalers e arrays."""
    X = df[['x1','x2','x3']].values.astype(float)
    y = df['y'].values.astype(float).reshape(-1,1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

# -----------------------
# Model building (Keras)
# -----------------------
def build_keras_model(input_dim=3, hidden_neurons=10, activation='relu'):
    model = Sequential([
        Dense(hidden_neurons, activation=activation, input_shape=(input_dim,)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -----------------------
# Train & evaluate
# -----------------------
def train_and_evaluate_keras(X_train, y_train, X_test, y_test, scaler_y, epochs=300, batch_size=32):
    tf.random.set_seed(42)
    model = build_keras_model()
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint(str(MODELS_DIR / "best_model.h5"), monitor='val_loss', save_best_only=True, verbose=0)
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
                        batch_size=batch_size, callbacks=[es, ckpt], verbose=2)
    # Predictions (inverse transform)
    y_pred_scaled = model.predict(X_test).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
    metrics = compute_metrics(y_true, y_pred)
    # save model
    model.save(MODELS_DIR / "final_model.h5")
    # save history plot
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss (treino vs validação)')
    plt.xlabel('Época'); plt.ylabel('MSE'); plt.legend()
    plt.savefig(PLOTS_DIR / "loss_curve.png", bbox_inches='tight'); plt.close()
    return metrics, y_true, y_pred, history

def train_and_evaluate_sklearn(X_train, y_train, X_test, y_test, scaler_y, max_iter=300):
    mlp = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                       max_iter=max_iter, early_stopping=True, validation_fraction=0.1,
                       n_iter_no_change=20, random_state=42, verbose=True)
    mlp.fit(X_train, y_train)
    y_pred_scaled = mlp.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
    metrics = compute_metrics(y_true, y_pred)
    # save model with joblib
    import joblib
    joblib.dump(mlp, MODELS_DIR / "mlp_model.joblib")
    # save loss curve if available
    if hasattr(mlp, "loss_curve_"):
        plt.figure()
        plt.plot(mlp.loss_curve_)
        plt.title("Loss (treino - iterações)")
        plt.xlabel("Iteração"); plt.ylabel("Loss")
        plt.savefig(PLOTS_DIR / "loss_curve.png", bbox_inches='tight'); plt.close()
    return metrics, y_true, y_pred, mlp

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "MAE": float(mae), "R2": float(r2)}

# -----------------------
# Plots and save results
# -----------------------
def save_common_plots(y_true, y_pred):
    # y_true vs y_pred
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx], linestyle='--', color='k')
    plt.xlabel("y_real"); plt.ylabel("y_previsto"); plt.title("y_real vs y_previsto")
    plt.savefig(PLOTS_DIR / "y_true_vs_y_pred.png", bbox_inches='tight'); plt.close()
    # residuals
    residuals = y_pred - y_true
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Distribuição dos resíduos (y_pred - y_true)"); plt.xlabel("Resíduo"); plt.ylabel("Frequência")
    plt.savefig(PLOTS_DIR / "residuals_hist.png", bbox_inches='tight'); plt.close()
    # save predictions
    preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": residuals})
    preds_df.to_csv(OUTDIR / "predictions.csv", index=False)
    return preds_df

def save_report(metadata, desc, corr, metrics):
    now = datetime.datetime.now().isoformat(timespec='seconds')
    report = {
        "generated_at": now,
        "metadata": metadata,
        "descriptive_stats_csv": str(OUTDIR / "descriptive_stats.csv"),
        "correlation_csv": str(OUTDIR / "correlation_matrix.csv"),
        "plots_dir": str(PLOTS_DIR),
        "models_dir": str(MODELS_DIR),
        "metrics": metrics
    }
    with open(OUTDIR / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # also brief textual report
    txt = (f"Relatório gerado em {now}\n\n"
           f"Métricas (conjunto de teste):\n"
           f" - MSE: {metrics['MSE']:.4f}\n"
           f" - MAE: {metrics['MAE']:.4f}\n"
           f" - R2:  {metrics['R2']:.4f}\n\n"
           "Veja os arquivos em output/ e as figuras em output/plots/\n")
    with open(OUTDIR / "report_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)

# -----------------------
# Main
# -----------------------
def main(args):
    # candidates for input files
    candidates = []
    if args.data:
        candidates.append(args.data)
    # common default paths
    candidates += [
        "./consumo_energia_full.csv",
        "./consumo_energia_train.csv",
        "./consumo_energia_test.csv",
        "./data/consumo_energia_full.csv",
        "./data/consumo_energia_train.csv",
        "./data/consumo_energia_test.csv"
    ]
    df_raw = auto_load_csv(candidates)
    df = detect_and_rename(df_raw)
    print("[i] Dimensões do dataset:", df.shape)
    # EDA
    desc, corr = do_eda(df)
    # preprocessing
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess(df, test_size=0.2, random_state=42)
    metadata = {"n_samples": int(df.shape[0]), "train_samples": int(X_train.shape[0]), "test_samples": int(X_test.shape[0]),
                "use_tensorflow": USE_TF}
    # train & evaluate
    if USE_TF:
        print("[i] TensorFlow detectado: treinando com Keras.")
        metrics, y_true, y_pred, _history = train_and_evaluate_keras(X_train, y_train, X_test, y_test, scaler_y,
                                                                    epochs=300, batch_size=32)
    else:
        print("[i] TensorFlow não disponível: usando sklearn MLPRegressor.")
        metrics, y_true, y_pred, _model = train_and_evaluate_sklearn(X_train, y_train, X_test, y_test, scaler_y, max_iter=300)
    # common plots
    preds_df = save_common_plots(y_true, y_pred)
    # save scalers
    import joblib
    joblib.dump(scaler_X, MODELS_DIR / "scaler_X.joblib")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y.joblib")
    # save EDA outputs were already saved in do_eda()
    save_report(metadata, desc, corr, metrics)
    # metrics CSV
    pd.DataFrame(list(metrics.items()), columns=["metric","value"]).to_csv(OUTDIR / "metrics.csv", index=False)
    print("[i] Done. Outputs em:", OUTDIR.resolve())
    print(" - plots:", PLOTS_DIR.resolve())
    print(" - modelos:", MODELS_DIR.resolve())
    print(" - resumo:", OUTDIR / "report_summary.txt")
    print(" - métricas:", OUTDIR / "metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina MLP para consumo energético.")
    parser.add_argument("--data", type=str, help="Caminho para CSV (opcional).")
    args = parser.parse_args()
    main(args)