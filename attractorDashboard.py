import streamlit as st
import pandas as pd
import numpy as np
import itertools
import graphviz
import plotly.express as px
import plotly.graph_objects as go
import random
import math
import json
import tensorflow as tf
import os
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ============================================================
# -------------------- MODEL DEFINITION ----------------------
# ============================================================

class CustomLSTM(tf.keras.Model):
    def __init__(self, inputSize, hiddenSize=None, numLayers=1,
                 dropout=None, bidirectional=None, activation=None, window=None, layer_configs=None):
        """
        A flexible model builder that accepts either scalar/list params or an explicit
        `layer_configs` list. Each entry in `layer_configs` is a dict with keys:
          - type: 'LSTM' | 'Conv1D' | 'Dense'
          - hiddenSize: int (units / filters)
          - dropout: float
          - bidirectional: bool (only for LSTM)
          - activation: string (e.g., 'relu','tanh',None)
          - kernel_size: int (for Conv1D)
        """
        super().__init__()

        # If explicit layer_configs provided, use it; otherwise fall back to older args
        if layer_configs:
            cfgs = list(layer_configs)
            numLayers = len(cfgs)
        else:
            # build simple configs from provided scalar/list args
            if isinstance(hiddenSize, (list, tuple)):
                sizes = list(hiddenSize)
            else:
                sizes = [hiddenSize or 32] * numLayers

            if isinstance(dropout, (list, tuple)):
                drops = list(dropout)
            else:
                drops = [dropout or 0.0] * numLayers

            if isinstance(bidirectional, (list, tuple)):
                bids = list(bidirectional)
            else:
                bids = [bidirectional or False] * numLayers

            cfgs = []
            for i in range(numLayers):
                cfgs.append({
                    'type': 'LSTM',
                    'hiddenSize': int(sizes[i]) if i < len(sizes) else int(sizes[-1]),
                    'dropout': float(drops[i]) if i < len(drops) else float(drops[-1]),
                    'bidirectional': bool(bids[i]) if i < len(bids) else bool(bids[-1]),
                    'activation': activation or 'Tanh',
                    'kernel_size': 3
                })

        layers_list = []
        for idx, lc in enumerate(cfgs):
            ltype = lc.get('type', 'LSTM')
            units = int(lc.get('hiddenSize', 32))
            dr = float(lc.get('dropout', 0.0))
            bid = bool(lc.get('bidirectional', False))
            act_name = lc.get('activation', activation)
            kernel = int(lc.get('kernel_size', 3))

            # Map activation strings to Keras activations (None -> linear)
            act = None
            if act_name in (None, 'None', 'linear'):
                act = None
            elif act_name == 'ReLU' or act_name == 'relu':
                act = 'relu'
            elif act_name == 'Tanh' or act_name == 'tanh':
                act = 'tanh'
            elif act_name == 'Sigmoid' or act_name == 'sigmoid':
                act = 'sigmoid'
            else:
                act = act_name

            return_seq = True if idx < (numLayers - 1) else False

            if ltype == 'LSTM':
                layer = tf.keras.layers.LSTM(units, return_sequences=return_seq, dropout=dr)
                if bid:
                    layer = tf.keras.layers.Bidirectional(layer)
                layers_list.append(layer)
            elif ltype == 'Conv1D':
                # Conv1D over time dimension; keep return_sequences by setting padding='same'
                conv = tf.keras.layers.Conv1D(filters=units, kernel_size=kernel, padding='same', activation=act)
                # If not the final feature-producing layer, ensure sequence output
                if return_seq:
                    layers_list.append(conv)
                else:
                    # final conv -> global pooling then flatten
                    layers_list.append(conv)
            elif ltype == 'Dense':
                # Dense applied via TimeDistributed to preserve timestep axis when needed
                if return_seq:
                    td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation=act))
                    layers_list.append(td)
                else:
                    layers_list.append(tf.keras.layers.Dense(units, activation=act))
            else:
                # Unknown type fallback to Dense
                if return_seq:
                    td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units, activation=act))
                    layers_list.append(td)
                else:
                    layers_list.append(tf.keras.layers.Dense(units, activation=act))

        # final output layer (regression)
        layers_list.append(tf.keras.layers.Dense(1, activation=None))

        self.model = tf.keras.Sequential(layers_list)

        # build the model so weights can be saved/loaded before first call
        if window is not None:
            self.model.build(input_shape=(None, window, inputSize))

    def call(self, x, training=False):
        return self.model(x, training=training)


# ============================================================
# ----------------- SEQUENCE GENERATOR -----------------------
# ============================================================

def createSequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        if not np.isnan(y[i + window]):
            Xs.append(X[i:i + window])
            ys.append(y[i + window])
    return np.array(Xs), np.array(ys)


def getSafeDevice():
    """Return a CUDA device if usable, otherwise fall back to CPU.

    Tries a tiny CUDA allocation/operation to detect runtime/compatibility
    issues (e.g. mismatched PyTorch/CUDA build vs GPU). If that fails we
    return CPU so the app continues instead of crashing with a CUDA kernel
    error.
    """
    # For TensorFlow: enable GPU memory growth if GPUs are present
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return 'gpu'
        except Exception:
            return 'cpu'
    return 'cpu'


# ============================================================
# --------------------- TRAINING WORKER ----------------------
# ============================================================

def trainModelRemote(column, df, attractionJson, config):
    # Select drivers based on attraction coefficient and threshold
    targetRows = [r for r in attractionJson if r["target_feature"] == column]
    targetRows = sorted(targetRows, key=lambda x: x["coefficient"], reverse=True)
    # Determine drivers based on threshold metric
    thr_metric = config.get('thresholdMetric', 'Rank')
    thr_value = config.get('attractionThreshold', 0.8)
    if thr_metric == 'Rank':
        topk = max(1, int(len(targetRows) * (1 - float(thr_value))))
        drivers = [r["driver"] for r in targetRows[:topk]]
    else:
        # Absolute: include drivers whose coefficient >= threshold
        drivers = [r["driver"] for r in targetRows if r.get('coefficient', 0) >= float(thr_value)]
        if len(drivers) == 0:
            # fallback to top-1 if none match
            drivers = [r["driver"] for r in targetRows[:1]]

    df = df.interpolate().bfill().ffill()

    X = df[drivers].values
    y = df[column].values

    Xseq, yseq = createSequences(X, y, config["window"])

    # Train / validation split
    split = int(len(Xseq) * 0.8)
    X_train, X_val = Xseq[:split], Xseq[split:]
    y_train, y_val = yseq[:split], yseq[split:]

    # Create TF datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32).reshape(-1, 1)))
    train_ds = train_ds.shuffle(buffer_size=1024).batch(config.get("batchSize")).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32).reshape(-1, 1)))
    val_ds = val_ds.batch(config.get("batchSize")).prefetch(tf.data.AUTOTUNE)

    # Build Keras model
    num_layers = config.get("numLayers") or len(config.get("layerConfigs", [])) or 1
    model = CustomLSTM(
        inputSize=len(drivers),
        layer_configs=config.get("layerConfigs", None),
        activation=config.get("activation", None),
        window=config["window"]
    )

    opt_cls = getattr(tf.keras.optimizers, config["optimizer"])
    optimizer = opt_cls(learning_rate=config.get("learningRate"))

    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(train_ds, validation_data=val_ds, epochs=config.get("epochs"), verbose=0)

    train_losses = history.history.get('loss', [])
    val_losses = history.history.get('val_loss', [])

    os.makedirs("models", exist_ok=True)
    # sanitize column name for filesystem and ensure .weights.h5 suffix
    def safeName(s):
        return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

    weights_fname = f"{safeName(column)}.weights.h5"
    model.save_weights(os.path.join("models", weights_fname))

    return {
        "column": column,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "drivers": drivers
    }


def evaluateConfig(column, df, attractionJson, baseConfig, trialCfg, quickEpochs=5):
    """Train with trialCfg (overrides) for a few epochs and return final validation loss."""
    # Build merged config
    cfg = dict(baseConfig)
    cfg.update(trialCfg)

    # reuse driver selection logic
    targetRows = [r for r in attractionJson if r["target_feature"] == column]
    targetRows = sorted(targetRows, key=lambda x: x["coefficient"], reverse=True)
    thr_metric = cfg.get('thresholdMetric', 'Rank')
    thr_value = cfg.get('attractionThreshold', 0.8)
    if thr_metric == 'Rank':
        topk = max(1, int(len(targetRows) * (1 - float(thr_value))))
        drivers = [r["driver"] for r in targetRows[:topk]]
    else:
        drivers = [r["driver"] for r in targetRows if r.get('coefficient', 0) >= float(thr_value)]
        if len(drivers) == 0:
            drivers = [r["driver"] for r in targetRows[:1]]

    df_local = df.interpolate().bfill().ffill()
    X = df_local[drivers].values
    y = df_local[column].values
    Xseq, yseq = createSequences(X, y, cfg["window"]) if len(X) > cfg["window"] else (np.zeros((0, cfg['window'], X.shape[1])), np.zeros((0,)))
    if len(Xseq) == 0:
        return float('inf')

    split = int(len(Xseq) * 0.8)
    X_train, X_val = Xseq[:split], Xseq[split:]
    y_train, y_val = yseq[:split], yseq[split:]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train.astype(np.float32).reshape(-1, 1)))
    train_ds = train_ds.shuffle(256).batch(cfg.get('batchSize', 32)).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val.astype(np.float32).reshape(-1, 1)))
    val_ds = val_ds.batch(cfg.get('batchSize', 32)).prefetch(tf.data.AUTOTUNE)

    # build model using layer_configs override if provided, otherwise use uniform layers
    if 'layer_configs' in cfg and cfg['layer_configs']:
        layer_cfgs = cfg['layer_configs']
    else:
        # uniform layers from trial or base
        nl = int(cfg.get('numLayers', 1))
        layer_cfgs = []
        for _ in range(nl):
            layer_cfgs.append({'type': 'LSTM', 'hiddenSize': int(cfg.get('hiddenSize', 32)), 'dropout': float(cfg.get('dropout', 0.0)), 'bidirectional': bool(cfg.get('bidirectional', False)), 'kernel_size': 3})

    model = CustomLSTM(inputSize=len(drivers), layer_configs=layer_cfgs, activation=cfg.get('activation', None), window=cfg['window'])
    opt_cls = getattr(tf.keras.optimizers, cfg.get('optimizer', 'Adam'))
    optimizer = opt_cls(learning_rate=float(cfg.get('learningRate', 1e-3)))
    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(train_ds, validation_data=val_ds, epochs=quick_epochs, verbose=0)
    val_losses = history.history.get('val_loss', [])
    return float(val_losses[-1]) if len(val_losses) else float('inf')


def hyperparameterSearch(column, df, attractionJson, baseConfig, searchOpts):
    """Simple random search over ranges in searchOpts; returns best trial and loss."""
    trials = int(searchOpts.get('trials', 10))
    best = {'loss': float('inf'), 'cfg': None}
    for t in range(trials):
        # sample params
        hid = int(random.randint(searchOpts.get('hidden_min', 16), searchOpts.get('hidden_max', 256)))
        drop = float(random.uniform(searchOpts.get('drop_min', 0.0), searchOpts.get('drop_max', 0.5)))
        lr = float(10 ** random.uniform(math.log10(searchOpts.get('lr_min', 1e-4)), math.log10(searchOpts.get('lr_max', 1e-2))))
        batch = int(random.choice(searchOpts.get('batch_choices', [16, 32, 64])))
        # optional params
        nl = int(random.randint(searchOpts.get('num_layers_min', baseConfig.get('num_layers', 1)), searchOpts.get('num_layers_max', baseConfig.get('num_layers', 1))))
        opt = random.choice(searchOpts.get('optimizer_choices', [baseConfig.get('optimizer', 'Adam')]))
        bid = random.choice(searchOpts.get('bidir_choices', [baseConfig.get('bidirectional', False)]))

        trialCfg = {'hiddenSize': hid, 'dropout': drop, 'learning_rate': lr, 'batch_size': batch, 'num_layers': nl, 'optimizer': opt, 'bidirectional': bid}
        loss = evaluateConfig(column, df, attractionJson, baseConfig, trialCfg, quickEpochs=searchOpts.get('quick_epochs', 5))
        if loss < best['loss']:
            best = {'loss': loss, 'cfg': dict(trialCfg)}
    return best


# ============================================================
# -------------------- STREAMLIT UI --------------------------
# ============================================================

st.set_page_config(layout="wide")
st.title("Attractors/ML Dashboard")
dataContainer = st.container(border=True)

uploaded = dataContainer.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    
    df = pd.read_csv(uploaded)
    df = df.drop(columns=[c for c in df.columns if c.lower()=="datetime"], errors="ignore")
    numericDf = df.select_dtypes(include=[np.number])

    # ========================================================
    # -------------------- DATA PREVIEW ----------------------
    # ========================================================

    dataContainer.subheader("Data Preview",help="First few rows of the numeric data. This gives a quick look at the values and can help identify any immediate issues such as incorrect parsing, unexpected missing values, or outliers. Note that only numeric columns are shown here since the attractor analysis relies on correlation metrics that require numeric data.")
    dataContainer.dataframe(numericDf.head())

    dataContainer.subheader("Missing Values per Column",help="Count of missing values in each column. High counts may indicate issues with data collection or the need for imputation strategies. Consider dropping columns with excessive missingness or using imputation techniques to fill in missing values based on the nature of the data and the analysis goals.")
    dataContainer.bar_chart(numericDf.isna().sum())

    dataContainer.subheader("Missing Data Heatmap",help="Visual representation of missing values in the dataset. Each row corresponds to a record and each column corresponds to a feature. Yellow indicates missing values, while blue indicates present values. This can help identify patterns of missingness, such as entire columns or rows that are missing data, which may inform data cleaning or imputation strategies.")
    fig_missing = px.imshow(numericDf.isna(), aspect="auto")
    dataContainer.plotly_chart(fig_missing)

    # Attractor controls (place before correlation computation so they affect method)
    with st.sidebar.expander("Attractor Config", expanded=True):
        st.caption("Configure how attractor drivers are selected based on correlation metrics and thresholds.")
        attractionMetric = st.selectbox("Attraction Metric", ["Pearson", "Spearman"], key='attractionMetric', help="Correlation method: 'Pearson' captures linear relationships, while 'Spearman' captures monotonic relationships based on rank. Spearman is more robust to outliers and non-linear but monotonic patterns.")
        thresholdMetric = st.selectbox("Threshold Metric", ["Rank", "Absolute"], key='thresholdMetric', help="Thresholding method: 'Rank' selects the top (1 - threshold) fraction of drivers based on their coefficient ranking, while 'Absolute' selects drivers whose coefficients exceed a fixed value. Rank is adaptive to the distribution of coefficients, while Absolute applies a fixed cutoff regardless of distribution.")
        attractionThreshold = st.slider("Attraction Threshold", 0.0, 1.0, 0.8, help="Threshold value for selecting drivers based on the chosen Threshold Metric. In 'Rank' mode, this represents the fraction of top drivers to select (e.g., 0.8 means select the top 20%). In 'Absolute' mode, this is the minimum coefficient value a driver must have to be selected.")

    # ========================================================
    # --------------- ATTRACTOR CALCULATION ------------------
    attractorContainer = st.container(border=True)
    # ========================================================
        
    
    attractorContainer.subheader("Correlation Heatmap",help="Correlation matrix of numeric features based on the selected Attraction Metric. This heatmap visualizes the pairwise relationships between features, where values close to 1 or -1 indicate strong positive or negative correlations, respectively. The selected Attraction Metric (Pearson or Spearman) determines how these correlations are calculated, which in turn affects the identification of attractor drivers for the predictive modeling.")
    # compute correlation according to selected metric
    try:
        corr_matrix = numericDf.corr(method='spearman') if attractionMetric == 'Spearman' else numericDf.corr()
    except Exception:
        corr_matrix = numericDf.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=False, width=800, height=800)
    fig_corr.update_xaxes(showticklabels=False)
    fig_corr.update_yaxes(showticklabels=False)
    attractorContainer.plotly_chart(fig_corr)
    results = []
    for c1, c2 in itertools.combinations(numericDf.columns, 2):
        results.append({
            "target_feature": c1,
            "driver": c2,
            "coefficient": abs(corr_matrix.loc[c1, c2])
        })

    resultsDf = pd.DataFrame(results)
    attractionJson = resultsDf.to_dict(orient="records")

    attractorContainer.subheader("Attraction Coefficient Distribution",help="Distribution of the absolute correlation coefficients between features, which represent the strength of the relationship between potential driver features and target features. The selection of drivers for the predictive model is based on these coefficients and the chosen thresholding method. The histogram is colored to indicate which coefficients are considered 'selected' based on the threshold, with a dashed red line indicating the cutoff value used for selection.")
    # Determine which coefficients are considered "selected" according to thresholdMetric
    if thresholdMetric == 'Rank':
        # select top (1 - threshold) fraction
        topk = max(1, int(len(resultsDf) * (1 - float(attractionThreshold))))
        # mark top-k after sorting
        ranked = resultsDf.sort_values('coefficient', ascending=False).reset_index(drop=True)
        ranked['selected'] = False
        ranked.loc[: topk-1, 'selected'] = True
        # merge selection back to original order
        resultsDf = resultsDf.merge(ranked[['coefficient', 'selected']], on='coefficient', how='left')
        # compute cutoff value for annotation
        cutoff_val = ranked.loc[topk-1, 'coefficient'] if len(ranked) >= topk else ranked['coefficient'].max()
        summary_text = f"Rank mode: top {topk} drivers selected. Cutoff coeff = {cutoff_val:.3f}"
    else:
        resultsDf['selected'] = resultsDf['coefficient'] >= float(attractionThreshold)
        cutoff_val = attractionThreshold
        summary_text = f"Absolute mode: coefficients >= {attractionThreshold:.3f} selected."
    selected_count = int(resultsDf['selected'].sum())
    total_count = len(resultsDf)
    attractorContainer.write(f"Selected drivers: {selected_count} / {total_count} ({selected_count/total_count:.1%})")

    fig_hist = px.histogram(resultsDf, x="coefficient", nbins=30, color='selected',
                            color_discrete_map={False: 'lightsteelblue', True: 'orange'},
                            labels={'selected': 'Selected'})
    fig_hist.update_layout(barmode='stack')
    try:
        fig_hist.add_vline(x=cutoff_val, line_dash='dash', line_color='red', annotation_text='Cutoff', annotation_position='top right')
    except Exception:
        pass
    attractorContainer.plotly_chart(fig_hist)
    attractorContainer.caption(summary_text)

    # ========================================================
    # ---------------- MODEL CONFIGURATION -------------------
    # ========================================================

    # Attempt to load a default NN config file before creating widgets
    default_cfg = None
    for p in (os.path.join('configs', 'config_default.json'), 'config_default.json'):
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    default_cfg = json.load(f)
                break
            except Exception:
                default_cfg = None
                break

    # If default config loaded, derive sensible defaults for the widget initial values
    def _g(g, camel, snake, default):
        if not g:
            return default
        return g.get(camel, g.get(snake, default))

    loaded_global = default_cfg.get('global', {}) if default_cfg else {}
    # Map loaded values (support both camelCase and snake_case saved files)
    default_hidden = int(_g(loaded_global, 'hiddenSize', 'hidden_size', 64))
    default_num_layers = int(_g(loaded_global, 'numLayers', 'num_layers', 2))
    default_dropout = float(_g(loaded_global, 'dropout', 'dropout', 0.2))
    default_bidir = bool(_g(loaded_global, 'bidirectional', 'bidirectional', False))
    default_lr = float(_g(loaded_global, 'learningRate', 'learning_rate', 1e-5))
    default_batch = int(_g(loaded_global, 'batchSize', 'batch_size', 32))
    default_window = int(_g(loaded_global, 'window', 'window', 20))
    default_epochs = int(_g(loaded_global, 'epochs', 'epochs', 10))
    default_optimizer = _g(loaded_global, 'optimizer', 'optimizer', 'Adam')
    default_activation = _g(loaded_global, 'activation', 'activation', 'ReLU')

    # If layer_configs present in the default, initialize session state accordingly (only if not already set)
    if default_cfg and 'layer_configs' in default_cfg and 'layer_configs' not in st.session_state:
        st.session_state['layer_configs'] = default_cfg.get('layer_configs', [])

    with st.sidebar.expander("Model Config", expanded=True):
        st.caption("Configure the neural network architecture and training parameters. You can also specify per-layer settings and perform auto-tuning for hyperparameter search.")
        hiddenSize = st.slider("Hidden Size", 16, 256, default_hidden, help="Number of units/filters in LSTM/Conv layers (if uniform) or base size for layer configs.")
        num_layers = st.number_input("Num Layers", 1, 8, default_num_layers, help="Number of layers in the model. If using uniform global settings, all layers will use the same hidden size, dropout, and bidirectionality. For per-layer customization, use the 'Per-layer Configuration' section below.")
        dropout = st.slider("Dropout", 0.0, 0.5, default_dropout, help="Dropout rate for LSTM layers (if uniform) or base dropout for layer configs.")
        bidirectional = st.checkbox("Bidirectional", value=default_bidir, help="Whether LSTM layers are bidirectional (if uniform) or base bidirectionality for layer configs.")
        _lr_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        _lr_labels = ["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"]
        _default_idx = _lr_options.index(1e-5) if 1e-5 in _lr_options else 2
        # choose default LR index based on default_lr from config file if present
        try:
            _default_idx = _lr_options.index(default_lr)
        except Exception:
            _default_idx = _lr_options.index(1e-5) if 1e-5 in _lr_options else 2
        learning_rate = float(st.selectbox('Learning Rate', options=_lr_options, format_func=lambda v: f"{v:.6g}", index=_default_idx, help="Choose learning rate by order-of-magnitude (significant figure)."))
        batch_size = st.slider("Batch Size", 16, 128, default_batch, help="Batch size for training.")
        window = st.slider("Window", 5, 50, default_window, help="Number of past timesteps to use as input for predicting the next value.")
        epochs = st.slider("Epochs", 5, 50, default_epochs, help="Number of training epochs.")
        # optimizer and activation widgets have explicit keys; prepopulate session_state values so selection reflects defaults
        if 'optimizer' not in st.session_state:
            st.session_state['optimizer'] = default_optimizer
        if 'activation' not in st.session_state:
            st.session_state['activation'] = default_activation
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD"], key='optimizer', help="Optimizer to use for training.")
        activation = st.selectbox("Activation", ["ReLU", "Identity", "Tanh"], key='activation', help="Activation function for LSTM layers (if uniform) or base activation for layer configs. 'Identity' means no activation (linear).")

    config = {
        "hiddenSize": hiddenSize,
        "numLayers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "attractionMetric": attractionMetric,
        "thresholdMetric": thresholdMetric,
        "attractionThreshold": attractionThreshold,
        "learningRate": learning_rate,
        "batchSize": batch_size,
        "window": window,
        "epochs": epochs,
        "optimizer": optimizer,
        "activation": activation
    }
    # defaults for optional tuning keys
    config.setdefault('autoTune', False)
    config.setdefault('tuneOpts', {})

    # ---------------- Per-layer configuration and commit -----------------
    # Initialize layer configs in session state if missing or if layer count changed
    if 'layer_configs' not in st.session_state or len(st.session_state.get('layer_configs', [])) != num_layers:
        # default each layer from current global settings
        st.session_state['layer_configs'] = [
            {
                'hiddenSize': hiddenSize,
                'dropout': dropout,
                'bidirectional': bidirectional
            } for _ in range(num_layers)
        ]

    with st.sidebar.expander('Per-layer Configuration', expanded=False):
        layer_idx = st.number_input('Layer to edit (1-indexed)', min_value=1, max_value=num_layers, value=1, step=1)
        li = layer_idx - 1

        # show current values for selected layer
        cur = st.session_state['layer_configs'][li]
        # layer type
        new_type = st.selectbox(f'Layer Type (Layer {layer_idx})', ['LSTM', 'Conv1D', 'Dense'], index=['LSTM', 'Conv1D', 'Dense'].index(cur.get('type', 'LSTM')), key=f'layer_type_{layer_idx}',help="Type of layer: 'LSTM' for recurrent layers, 'Conv1D' for temporal convolutional layers, and 'Dense' for fully connected layers. This determines the operations performed at this layer and the meaning of other parameters (e.g., 'hiddenSize' is units for LSTM/Dense but filters for Conv1D).")
        new_units = st.slider(f'Units/Filters (Layer {layer_idx})', 4, 1024, int(cur.get('hiddenSize', hiddenSize)), help="Number of units for LSTM/Dense or filters for Conv1D.")
        new_dropout = st.slider(f'Dropout (Layer {layer_idx})', 0.0, 0.9, float(cur.get('dropout', dropout)),help="Dropout rate for LSTM layers. Ignored for Conv1D and Dense layers.")
        new_bidir = st.checkbox(f'Bidirectional (Layer {layer_idx})', value=bool(cur.get('bidirectional', bidirectional)),help="Whether the layer is bidirectional (only applicable for LSTM layers). Ignored for Conv1D and Dense layers." )
        new_kernel = None
        if new_type == 'Conv1D':
            new_kernel = st.slider(f'Kernel Size (Layer {layer_idx})', 1, 11, int(cur.get('kernel_size', 3)))

        if st.button(f'Commit Layer {layer_idx}'):
            entry = {
                'type': new_type,
                'hiddenSize': int(new_units),
                'dropout': float(new_dropout),
                'bidirectional': bool(new_bidir)
            }
            if new_kernel is not None:
                entry['kernel_size'] = int(new_kernel)
            st.session_state['layer_configs'][li] = entry
            st.success(f'Layer {layer_idx} committed')

        if st.button('Reset Layer Configs'):
            st.session_state['layer_configs'] = [
                {
                    'type': 'LSTM',
                    'hiddenSize': hiddenSize,
                    'dropout': dropout,
                    'bidirectional': bidirectional,
                    'kernel_size': 3
                } for _ in range(num_layers)
            ]
            st.info('Layer configs reset')

    # expose the layer_configs to the training config (camelCase keys)
    config['layerConfigs'] = st.session_state['layer_configs']
    config['layerSizes'] = [int(lc.get('hiddenSize', 32)) for lc in st.session_state['layer_configs']]
    config['layerDropouts'] = [float(lc.get('dropout', 0.0)) for lc in st.session_state['layer_configs']]
    config['layerBidir'] = [bool(lc.get('bidirectional', False)) for lc in st.session_state['layer_configs']]
    # ---------------- Auto-tune options -----------------
    with st.sidebar.expander('Auto-tune (Hyperparameter Search)', expanded=False):
        autoTune = st.checkbox('Enable Auto-tune')
        if autoTune:
            st.warning('Auto-tune is enabled. This will override manual hyperparameter settings.')
            tune_trials = st.number_input('Trials', min_value=1, max_value=200, value=20, step=1, help="Number of random hyperparameter configurations to try during auto-tuning. More trials increase the chance of finding a better configuration but take more time.")
            tune_quick_epochs = st.number_input('Quick epochs per trial', min_value=1, max_value=20, value=5, step=1, help="Number of epochs to train each trial configuration during auto-tuning. This is a quick evaluation to estimate performance; it should be low to keep tuning time reasonable, but high enough to get a meaningful signal.")
            tune_hidden_min = st.number_input('Hidden size min', min_value=4, max_value=1024, value=16, step=1, help="Minimum hidden size (units/filters) to consider during auto-tuning. The tuning process will sample hidden sizes between this minimum and the maximum specified below.")
            tune_hidden_max = st.number_input('Hidden size max', min_value=4, max_value=1024, value=128, step=1, help="Maximum hidden size (units/filters) to consider during auto-tuning. The tuning process will sample hidden sizes between the minimum and this maximum.")
            tune_drop_min = st.slider('Dropout min', 0.0, 0.9, 0.0, help="Minimum dropout rate to consider during auto-tuning. The tuning process will sample dropout rates between this minimum and the maximum specified below.")
            tune_drop_max = st.slider('Dropout max', 0.0, 0.9, 0.5, help="Maximum dropout rate to consider during auto-tuning. The tuning process will sample dropout rates between the minimum and this maximum.")
            tune_lr_min = st.number_input('LR min', value=1e-4, format="%.6f", min_value=1e-6, max_value=1e-1, step=1e-5, help="Minimum learning rate to consider during auto-tuning. The tuning process will sample learning rates on a log scale between this minimum and the maximum specified below.")
            tune_lr_max = st.number_input('LR max', value=1e-2, format="%.6f", min_value=1e-6, max_value=1e-1, step=1e-5, help="Maximum learning rate to consider during auto-tuning. The tuning process will sample learning rates on a log scale between the minimum and this maximum.")
            tune_batch_choices = st.multiselect('Batch sizes', [16, 32, 64, 128], default=[32], help="Batch sizes to consider during auto-tuning. The tuning process will randomly select from these batch sizes for each trial configuration.")

            config['autoTune'] = autoTune
            config['tuneOpts'] = {
                'trials': int(tune_trials),
                'quick_epochs': int(tune_quick_epochs),
                'hidden_min': int(tune_hidden_min),
                'hidden_max': int(tune_hidden_max),
                'drop_min': float(tune_drop_min),
                'drop_max': float(tune_drop_max),
                'lr_min': float(tune_lr_min),
                'lr_max': float(tune_lr_max),
                'batch_choices': tune_batch_choices or [32]
            }
        # extendable search options
        config.setdefault('tuneOpts', {})
        config['tuneOpts'].update({
            'num_layers_min': 1,
            'num_layers_max': max(1, num_layers + 2),
            'optimizer_choices': ['Adam', 'SGD'],
            'bidir_choices': [True, False]
        })

    # ----------------- Save / Load NN configurations -----------------
    with st.sidebar.expander('Save / Load NN Config', expanded=False):
        cfgName = st.text_input('Config name (for save)', value=f"config_")
        if st.button('Save NN Config'):
            os.makedirs('configs', exist_ok=True)
            save_path = os.path.join('configs', f"{cfgName}.json")
            to_save = {
                'meta': {
                    'saved_at': datetime.utcnow().isoformat() + 'Z',
                    'name': cfgName
                },
                'global': {
                    'hiddenSize': hiddenSize,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'bidirectional': bidirectional,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'window': window,
                    'epochs': epochs,
                    'optimizer': optimizer,
                    'activation': activation
                },
                'layer_configs': st.session_state.get('layer_configs', [])
            }
            with open(save_path, 'w') as f:
                json.dump(to_save, f, indent=2)
            st.success(f"Saved {save_path}")

        # Load existing configs
        os.makedirs('configs', exist_ok=True)
        saved = [f for f in os.listdir('configs') if f.endswith('.json')]
        selectedCfg = st.selectbox('Load saved config', [''] + saved)
        if st.button('Load NN Config'):
            if selectedCfg:
                with open(os.path.join('configs', selectedCfg), 'r') as f:
                    loaded = json.load(f)
                # apply loaded global values and layer_configs
                g = loaded.get('global', {})
                st.session_state['layer_configs'] = loaded.get('layer_configs', st.session_state.get('layer_configs', []))
                # attempt to set globals (won't rewrite widgets state, but provide feedback)
                st.success(f"Loaded {selectedCfg}")
    # ====================================================
    # ------------- Real-Time NN Diagram -----------------
    MLContainer = st.container(border=True)
    # ====================================================
    

    MLContainer.subheader("Neural Network Architecture",help="Visualization of the neural network architecture based on the current configuration. This diagram updates in real-time as you modify the model settings, including the number of layers, hidden sizes, dropout rates, and bidirectionality. Each layer box displays its type and parameters, and arrows indicate the flow of data from input to output.")

    directionMultiplier = 2 if bidirectional else 1

    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')

    dot.node("Input",
             f"Input\nFeatures: {numericDf.shape[1]}\nWindow: {window}",
             shape="box")

    previous = "Input"

    for i in range(num_layers):
        name = f"LSTM_{i+1}"
        cfg = st.session_state['layer_configs'][i] if 'layer_configs' in st.session_state and i < len(st.session_state['layer_configs']) else {'hiddenSize': hiddenSize, 'dropout': dropout, 'bidirectional': bidirectional}
        dot.node(name,
              f"{cfg.get('type','LSTM')} {i+1}\nHidden: {cfg.get('hiddenSize')}\nDropout: {cfg.get('dropout')}\nBidir: {cfg.get('bidirectional')}",
              shape="box")
        dot.edge(previous, name)
        previous = name

    dot.node("Output", f"Dense\nActivation: {activation}", shape="box")
    dot.edge(previous, "Output")

    MLContainer.graphviz_chart(dot)

    # ========================================================
    # ---------------- TRAINING ------------------------------
    # ========================================================

    targets = sorted(resultsDf["target_feature"].unique())
    selected_targets = MLContainer.multiselect("Select Columns to Train", targets)

    if MLContainer.button("Start Training"):

        resultsContainer = []
        total = len(selected_targets)
        if total == 0:
            MLContainer.info("No targets selected for training.")
        else:
            progress = MLContainer.progress(0)
            status_boxes = {col: MLContainer.empty() for col in selected_targets}

            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(trainModelRemote, col, numericDf, attractionJson, config): col for col in selected_targets}

                completed = 0
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        res = future.result()
                        resultsContainer.append(res)
                        status_boxes[col].success(f"Trained: {col}")
                    except Exception as e:
                        status_boxes[col].error(f"Failed: {col} — {e}")
                    completed += 1
                    progress.progress(int(completed / total * 100))

            MLContainer.success("Training completed.")
            MLContainer.write(f"Trained {len(resultsContainer)} / {total} models.")
            st.session_state["trained_models"] = resultsContainer

    # Auto-tune trigger
    if config.get('autoTune') and MLContainer.button('Start Auto-tune'):
        if len(selected_targets) == 0:
            MLContainer.info('Select at least one target column to auto-tune.')
        else:
            tuneResults = {}
            progress = MLContainer.progress(0)
            for i, col in enumerate(selected_targets):
                best = hyperparameterSearch(col, numericDf, attractionJson, config, config.get('tuneOpts', {}))
                tuneResults[col] = best
                progress.progress(int((i + 1) / len(selected_targets) * 100))
            st.session_state['tuneResults'] = tuneResults
            MLContainer.success('Auto-tune complete')
            MLContainer.write(tuneResults)

    # ========================================================
    # --------------------- SHOW RESULTS ---------------------
    # ========================================================

    if "trained_models" in st.session_state:
        for res in st.session_state["trained_models"]:
            MLContainer.subheader(f"Training Curves: {res['column']}",help="Training and validation loss curves for each trained model. These plots show how well the model is learning over time, with lower loss values indicating better performance. If the validation loss starts to increase while training loss continues to decrease, it may indicate overfitting.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=res["train_losses"], name="Train"))
            fig.add_trace(go.Scatter(y=res["val_losses"], name="Validation"))
            # ensure y-axis always includes zero as the baseline
            fig.update_yaxes(rangemode='tozero')

            MLContainer.plotly_chart(fig)

            MLContainer.write(f"Drivers Used:")
            MLContainer.write(res['drivers'])


    # ========================================================
    # ---------------- INFILL / FORECAST ---------------------
    # ========================================================

    MLContainer.subheader("Infill / Forecast",help="Generate infilled and forecasted versions of the dataset. This process fills missing values in selected target columns using trained models and extends the dataset by forecasting future values.")

    forecast_steps = MLContainer.slider("Forecast Steps", 0, 1000, 0)

    if MLContainer.button("Generate Infilled Dataset"):

        # start from a copy without globally filling missing values
        dfFilled = numericDf.copy()

        # Determine which selected targets have trained models
        trained_models = st.session_state.get("trained_models", [])
        trained_columns = {m["column"]: m for m in trained_models}

        to_infill = [c for c in selected_targets if c in trained_columns]
        skipped = [c for c in selected_targets if c not in trained_columns]

        if skipped:
            MLContainer.warning(f"Skipping untrained columns: {', '.join(skipped)}")

        if not to_infill:
            MLContainer.info("No trained columns available to infill.")
        else:
            for column in to_infill:
                # try to find drivers length from trained model metadata
                meta = trained_columns.get(column, {})
                drivers = meta.get("drivers") or []
                inputSize = len(drivers) if drivers else (len(numericDf.columns) - 1)

                # create model matching training input size
                model = CustomLSTM(
                    inputSize=inputSize,
                    layer_configs=config.get('layer_configs', None),
                    activation=activation,
                    window=window
                )

                # load Keras weights
                def safeName(s):
                    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

                weights_fname = f"{safeName(column)}.weights.h5"
                weights_path = os.path.join("models", weights_fname)

                # ensure model is built before loading weights
                try:
                    model.build((None, window, inputSize))
                except Exception:
                    try:
                        dummy = np.zeros((1, window, inputSize), dtype=np.float32)
                        _ = model(tf.convert_to_tensor(dummy))
                    except Exception:
                        pass

                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                model.trainable = False

                # Simplified infill logic for trained column
                dfFilled[column] = dfFilled[column].fillna(dfFilled[column].mean())

        # Forecasting (naive extension)
        for i in range(forecast_steps):
            dfFilled.loc[len(dfFilled)] = dfFilled.iloc[-1]

        st.success("Infilled / Forecasted dataset ready.")

        st.download_button(
            "Download Completed Dataset",
            dfFilled.to_csv(index=False),
            file_name="completed_dataset.csv"
        )