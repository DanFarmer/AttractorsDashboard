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
import time

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

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=quickEpochs, verbose=0)
    train_time_s = time.time() - t0

    val_losses = history.history.get('val_loss', [])
    final_val_loss = float(val_losses[-1]) if len(val_losses) else float('inf')

    # Count total trainable parameters as a proxy for model size
    try:
        model_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    except Exception:
        model_params = 0

    return {
        'valLoss': final_val_loss,
        'trainTimeS': float(train_time_s),
        'modelParams': model_params
    }


def hyperparameterSearch(column, df, attractionJson, baseConfig, searchOpts):
    """Simple random search over ranges in searchOpts; returns best trial and loss."""
    trials = int(searchOpts.get('trials', 10))
    best = {'loss': float('inf'), 'cfg': None}
    trial_records = []
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

        trialCfg = {
            'hiddenSize': hid,
            'dropout': drop,
            'learningRate': lr,
            'batchSize': batch,
            'numLayers': nl,
            'optimizer': opt,
            'bidirectional': bid
        }
        eval_result = evaluateConfig(column, df, attractionJson, baseConfig, trialCfg, quickEpochs=searchOpts.get('quick_epochs', 5))
        loss = eval_result.get('valLoss', float('inf')) if isinstance(eval_result, dict) else float(eval_result)
        train_time_s = eval_result.get('trainTimeS', 0.0) if isinstance(eval_result, dict) else 0.0
        model_params = eval_result.get('modelParams', 0) if isinstance(eval_result, dict) else 0
        trial_records.append({
            'trial': t + 1,
            'hiddenSize': hid,
            'dropout': drop,
            'learningRate': lr,
            'batchSize': batch,
            'numLayers': nl,
            'optimizer': opt,
            'bidirectional': bool(bid),
            'valLoss': float(loss),
            'trainTimeS': float(train_time_s),
            'modelParams': int(model_params)
        })
        if loss < best['loss']:
            best = {'loss': loss, 'cfg': dict(trialCfg)}
    return {
        'best': best,
        'trials': trial_records
    }


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
    missing_sort_mode = dataContainer.radio(
        "Missing-value bar order",
        ["Dataframe order", "Ascending missing count"],
        horizontal=True,
        key='missing_sort_mode'
    )
    missing_counts = numericDf.isna().sum()
    if missing_sort_mode == "Ascending missing count":
        missing_counts = missing_counts.sort_values(ascending=True)
    else:
        missing_counts = missing_counts.reindex(numericDf.columns)

    missing_plot_df = pd.DataFrame({
        'column': missing_counts.index,
        'missing_count': missing_counts.values
    })
    fig_missing_bar = px.bar(missing_plot_df, x='column', y='missing_count', labels={'column': 'Column', 'missing_count': 'Missing Values'})
    fig_missing_bar.update_xaxes(categoryorder='array', categoryarray=missing_plot_df['column'].tolist())
    dataContainer.plotly_chart(fig_missing_bar, use_container_width=True)

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

    attractorContainer.subheader("Attraction Coefficient Chord Diagram",help="A circular chord-style diagram visualizing feature relationships from absolute correlation coefficients. Nodes are arranged around a circle and curved interior links represent pairwise relationships; thicker and darker chords indicate stronger relationships.")
    # Plotly graph_objects does not provide go.Chord; draw a circular chord-style chart using curved line traces.
    feature_names = corr_matrix.columns.tolist()
    node_idx = {name: i for i, name in enumerate(feature_names)}

    # Build links from upper-triangle feature pairs
    pair_links = []
    for c1, c2 in itertools.combinations(feature_names, 2):
        coef = float(abs(corr_matrix.loc[c1, c2]))
        if not np.isnan(coef) and coef > 0:
            pair_links.append((c1, c2, coef))

    # Keep the diagram readable for larger datasets by limiting to strongest links
    max_links = 120
    pair_links = sorted(pair_links, key=lambda x: x[2], reverse=True)#[:max_links]
    pair_links = [entry for entry in pair_links if entry[2] > attractionThreshold]

    if pair_links:
        n_nodes = len(feature_names)
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        radius = 1.0
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)

        # map name -> (x,y)
        pos = {name: (node_x[idx], node_y[idx]) for name, idx in node_idx.items()}

        # chord widths/opacities scaled by coefficient strength
        vals = np.array([v for _, _, v in pair_links], dtype=float)
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = (vmax - vmin) if vmax > vmin else 1.0

        fig_rel = go.Figure()

        # Draw curved links (quadratic Bezier via center control)
        for src, tgt, val in pair_links:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]

            t = np.linspace(0, 1, 24)
            # control point at center creates an inward "chord" arc
            cx, cy = 0.0, 0.0
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1

            strength = (val - vmin) / denom
            width = 0.8 + 4.0 * strength
            alpha = 0.15 + 0.55 * strength

            fig_rel.add_trace(go.Scatter(
                x=bx,
                y=by,
                mode='lines',
                hoverinfo='text',
                text=[f"{src} ↔ {tgt}<br>|corr|={val:.3f}"] * len(bx),
                line=dict(color=f'rgba(30, 120, 180, {alpha:.3f})', width=width),
                showlegend=False
            ))

        # Draw outer circle guide
        circle_t = np.linspace(0, 2 * np.pi, 300)
        fig_rel.add_trace(go.Scatter(
            x=1.03 * np.cos(circle_t),
            y=1.03 * np.sin(circle_t),
            mode='lines',
            line=dict(color='rgba(80,80,80,0.35)', width=1),
            hoverinfo='skip',
            showlegend=False
        ))

        # Draw nodes and labels
        fig_rel.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=feature_names,
            textposition='middle center',
            marker=dict(size=10, color='midnightblue'),
            hoverinfo='text',
            hovertext=[f"Feature: {n}" for n in feature_names],
            showlegend=False
        ))

        fig_rel.update_layout(
            title="Feature Correlation Chord Diagram",
            width=900,
            height=900,
            xaxis=dict(visible=False, scaleanchor='y', scaleratio=1),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        attractorContainer.plotly_chart(fig_rel, use_container_width=True)
    else:
        attractorContainer.info("No non-zero relationships available to plot.")
    
    

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
                search_out = hyperparameterSearch(col, numericDf, attractionJson, config, config.get('tuneOpts', {}))
                tuneResults[col] = search_out
                progress.progress(int((i + 1) / len(selected_targets) * 100))
            st.session_state['tuneResults'] = tuneResults
            MLContainer.success('Auto-tune complete')
            MLContainer.write({k: v.get('best') for k, v in tuneResults.items()})

    # Auto-tune diagnostics visualization
    if 'tuneResults' in st.session_state and st.session_state['tuneResults']:
        MLContainer.subheader("Auto-tune Architecture Performance", help="Parallel-coordinates view of all hyperparameter trials from auto-tune. Each polyline is one trial. Use the Highlight control to emphasise the best model by a given criterion. Axes include training time and total parameter count so architectural trade-offs are visible.")
        tuned_targets = sorted(list(st.session_state['tuneResults'].keys()))
        chosen_tuned_target = MLContainer.selectbox("Target for Auto-tune Plot", tuned_targets, key='tuned_target_for_parallel')

        highlight_mode = MLContainer.radio(
            "Highlight",
            ["Lowest Validation Loss", "Fastest Training", "Smallest Model", "None"],
            horizontal=True,
            key='tune_highlight_mode',
            help="Dim all trials except the one that wins the selected criterion."
        )

        chosen_payload = st.session_state['tuneResults'].get(chosen_tuned_target, {})
        trial_rows = chosen_payload.get('trials', []) if isinstance(chosen_payload, dict) else []

        if trial_rows:
            tune_df = pd.DataFrame(trial_rows)
            # ensure new columns exist for runs recorded before this version
            for _col, _default in [('trainTimeS', 0.0), ('modelParams', 0)]:
                if _col not in tune_df.columns:
                    tune_df[_col] = _default
            tune_df = tune_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['valLoss'])

            if not tune_df.empty:
                # Encode categorical options for parallel-coordinates numeric axes
                optimizer_values = sorted(tune_df['optimizer'].dropna().unique().tolist())
                optimizer_map = {name: idx for idx, name in enumerate(optimizer_values)}
                tune_df['optimizerCode'] = tune_df['optimizer'].map(optimizer_map).astype(float)
                tune_df['bidirectionalCode'] = tune_df['bidirectional'].astype(int)
                tune_df['log10LearningRate'] = np.log10(tune_df['learningRate'].clip(lower=1e-12))

                # Determine highlight index
                highlight_idx = None
                if highlight_mode == "Lowest Validation Loss":
                    highlight_idx = int(tune_df['valLoss'].idxmin())
                elif highlight_mode == "Fastest Training":
                    highlight_idx = int(tune_df['trainTimeS'].idxmin())
                elif highlight_mode == "Smallest Model":
                    highlight_idx = int(tune_df['modelParams'].idxmin())

                # Build a line opacity column: 1.0 for highlighted, 0.08 for others
                if highlight_idx is not None:
                    tune_df['_opacity'] = 0.08
                    tune_df['_width'] = 1
                    tune_df.loc[highlight_idx, '_opacity'] = 1.0
                else:
                    tune_df['_opacity'] = 1.0

                # For highlight mode: colour = opacity bucket (1 = bright, 0 = dim)
                tune_df['_colorKey'] = tune_df['_opacity'].apply(lambda v: 1.0 if v == 1.0 else 0.0)
                color_col = '_colorKey' if highlight_idx is not None else 'valLoss'
                color_scale = (
                    [[0.0, 'rgba(100,100,100,0.01)'], [1.0, 'rgba(255,255,0,2)']]
                    if highlight_idx is not None
                    else px.colors.sequential.Viridis_r
                )

                _labels = {
                    'hiddenSize': 'Hidden Size',
                    'dropout': 'Dropout',
                    'numLayers': 'Num Layers',
                    'batchSize': 'Batch Size',
                    'log10LearningRate': 'log10(LR)',
                    'bidirectionalCode': 'Bidirectional',
                    'optimizerCode': 'Optimizer',
                    'valLoss': 'Val Loss',
                    'trainTimeS': 'Train Time (s)',
                    'modelParams': 'Model Params',
                }

                fig_parallel = px.parallel_coordinates(
                    tune_df,
                    dimensions=[
                        'hiddenSize',
                        'numLayers',
                        'dropout',
                        'batchSize',
                        'log10LearningRate',
                        'bidirectionalCode',
                        'optimizerCode',
                        'trainTimeS',
                        'modelParams',
                        'valLoss'
                    ],
                    color=color_col,
                    color_continuous_scale=color_scale,
                    labels=_labels
                )
                fig_parallel.update_layout(
                    height=580,
                    coloraxis_showscale=(highlight_idx is None)
                )
                MLContainer.plotly_chart(fig_parallel, use_container_width=True)

                if optimizer_map:
                    optimizer_legend = ", ".join([f"{v} = {k}" for k, v in optimizer_map.items()])
                    MLContainer.caption(f"Optimizer axis codes: {optimizer_legend}")

                # Summary callout for highlighted trial
                if highlight_idx is not None:
                    h = tune_df.loc[highlight_idx]
                    MLContainer.info(
                        f"**Highlighted trial #{int(h['trial'])}** — "
                        f"Val Loss: {h['valLoss']:.5f} | "
                        f"Train Time: {h['trainTimeS']:.2f}s | "
                        f"Params: {int(h['modelParams']):,} | "
                        f"Hidden: {int(h['hiddenSize'])} | "
                        f"Layers: {int(h['numLayers'])} | "
                        f"LR: {h['learningRate']:.2e} | "
                        f"Optimizer: {h['optimizer']}"
                    )
                else:
                    best_payload = chosen_payload.get('best', {}) if isinstance(chosen_payload, dict) else {}
                    MLContainer.write(f"Best validation loss for **{chosen_tuned_target}**: `{best_payload.get('loss', float('nan')):.5f}`")
            else:
                MLContainer.info("No valid auto-tune trials available to plot.")
        else:
            MLContainer.info("No trial history found for this target. Run Auto-tune to generate trial data.")

        

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

    forecast_steps = MLContainer.slider("Forecast Steps", 0, 1000, 0)## forecast_steps can be set to 0 to disable forecasting and only perform infilling

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
            # Number of MC-dropout forward passes for uncertainty estimation
            N_MC = 20

            def _safeName(s):
                return re.sub(r'[^A-Za-z0-9._-]+', '_', s)

            def _load_model(column, drivers, inputSize):
                m = CustomLSTM(
                    inputSize=inputSize,
                    layer_configs=config.get('layerConfigs', None),
                    activation=activation,
                    window=window
                )
                try:
                    m.build((None, window, inputSize))
                except Exception:
                    try:
                        dummy = np.zeros((1, window, inputSize), dtype=np.float32)
                        _ = m(tf.convert_to_tensor(dummy))
                    except Exception:
                        pass
                wp = os.path.join("models", f"{_safeName(column)}.weights.h5")
                if os.path.exists(wp):
                    m.load_weights(wp)
                m.trainable = False
                return m

            def _mc_predict(model, batch_tensor, n_mc):
                """Run N MC-dropout passes; return (mean, std) arrays of shape (B,)."""
                passes = np.stack([
                    model(batch_tensor, training=True).numpy().squeeze(-1)
                    for _ in range(n_mc)
                ])  # (N_MC, B)
                return passes.mean(axis=0), passes.std(axis=0)

            # Interpolate driver columns so predictions have clean inputs
            df_interp = dfFilled.copy()
            for col in numericDf.columns:
                df_interp[col] = df_interp[col].interpolate().bfill().ffill()

            infill_plots = {}  # column -> plot-data dict

            for column in to_infill:
                meta = trained_columns.get(column, {})
                drivers = meta.get("drivers") or []
                inputSize = len(drivers) if drivers else max(1, len(numericDf.columns) - 1)
                other_cols = [c for c in numericDf.columns if c != column]

                # Record original missing mask
                original_missing = numericDf[column].isna().values.copy()

                model = _load_model(column, drivers, inputSize)

                # Build driver array from interpolated copy
                X_all = (df_interp[drivers].values if drivers else df_interp[other_cols].values).astype(np.float32)
                n = len(X_all)

                # Identify all positions where we have a complete window
                valid_positions = [
                    i for i in range(n - window)
                    if not np.any(np.isnan(X_all[i:i + window]))
                ]

                pred_mean = np.full(n, np.nan)
                pred_std  = np.full(n, np.nan)

                if valid_positions:
                    batch_X = np.stack([X_all[i:i + window] for i in valid_positions])
                    means, stds = _mc_predict(model, tf.convert_to_tensor(batch_X), N_MC)
                    for k, pos in enumerate(valid_positions):
                        pred_mean[pos + window] = float(means[k])
                        pred_std[pos + window]  = float(stds[k])

                # Apply predictions to fill missing positions
                y_vals = dfFilled[column].values.copy().astype(float)
                infilled_indices = []
                for i in range(n):
                    if original_missing[i] and not np.isnan(pred_mean[i]):
                        y_vals[i] = pred_mean[i]
                        infilled_indices.append(i)
                dfFilled[column] = y_vals

                infill_plots[column] = {
                    'original':        numericDf[column].values.copy().astype(float),
                    'filled':          y_vals,
                    'pred_mean':       pred_mean,
                    'pred_std':        pred_std,
                    'missing_mask':    original_missing,
                    'n_original':      n,
                    'drivers':         drivers,
                    'inputSize':       inputSize,
                    'other_cols':      other_cols,
                }

            # ---- Forecasting ----
            n_original = len(dfFilled)
            forecast_records = {}  # column -> {'means': [], 'stds': []}

            for column in to_infill:
                pdata = infill_plots[column]
                drivers   = pdata['drivers']
                inputSize = pdata['inputSize']
                other_cols = pdata['other_cols']

                if forecast_steps <= 0:
                    continue

                model = _load_model(column, drivers, inputSize)

                # Seed rolling window from filled data
                seed_src = dfFilled[drivers].values if drivers else dfFilled[other_cols].values
                rolling = seed_src[-window:].astype(np.float32).copy()

                fc_means, fc_stds = [], []
                for _ in range(forecast_steps):
                    t = tf.convert_to_tensor(rolling[np.newaxis])
                    m_val, s_val = _mc_predict(model, t, N_MC)
                    fc_means.append(float(m_val[0]))
                    fc_stds.append(float(s_val[0]))
                    # Advance rolling window: use predicted value for first feature slot
                    next_row = rolling[-1].copy()
                    next_row[0] = float(m_val[0])
                    rolling = np.vstack([rolling[1:], next_row[np.newaxis]])

                forecast_records[column] = {'means': fc_means, 'stds': fc_stds}

            # Extend dfFilled rows for forecast
            if forecast_steps > 0:
                last_row = dfFilled.iloc[-1].copy()
                ext = pd.DataFrame([last_row] * forecast_steps)
                ext.index = range(n_original, n_original + forecast_steps)
                dfFilled = pd.concat([dfFilled, ext])
                for column in to_infill:
                    if column in forecast_records:
                        for i, fm in enumerate(forecast_records[column]['means']):
                            dfFilled.loc[n_original + i, column] = fm

            # ---- Plot each infilled column ----
            for column in to_infill:
                pdata = infill_plots.get(column)
                if pdata is None:
                    continue

                orig         = pdata['original']
                missing_mask = pdata['missing_mask']
                pred_mean    = pdata['pred_mean']
                pred_std     = pdata['pred_std']
                n_orig       = pdata['n_original']
                x_all        = list(range(n_orig))

                fig = go.Figure()

                # 1. Observed (non-missing) values
                known_x = [i for i in x_all if not missing_mask[i]]
                known_y = [orig[i] for i in known_x]
                fig.add_trace(go.Scatter(
                    x=known_x, y=known_y,
                    mode='lines',
                    name='Observed',
                    line=dict(color='steelblue', width=2)
                ))

                # 2. Infilled values with 95 % CI band
                infill_x = [i for i in x_all if missing_mask[i] and not np.isnan(pred_mean[i])]
                if infill_x:
                    inf_y     = [pred_mean[i] for i in infill_x]
                    inf_upper = [pred_mean[i] + 1.96 * pred_std[i] for i in infill_x]
                    inf_lower = [pred_mean[i] - 1.96 * pred_std[i] for i in infill_x]

                    fig.add_trace(go.Scatter(
                        x=infill_x + infill_x[::-1],
                        y=inf_upper + inf_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 140, 0, 0.20)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Infill 95% CI',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=infill_x, y=inf_y,
                        mode='markers',
                        name='Infilled',
                        marker=dict(color='darkorange', size=6, symbol='circle')
                    ))

                # 3. Forecast with 95 % CI band
                if column in forecast_records and forecast_steps > 0:
                    fc = forecast_records[column]
                    fc_x     = list(range(n_orig, n_orig + forecast_steps))
                    fc_means_arr = fc['means']
                    fc_stds_arr  = fc['stds']
                    fc_upper = [m + 1.96 * s for m, s in zip(fc_means_arr, fc_stds_arr)]
                    fc_lower = [m - 1.96 * s for m, s in zip(fc_means_arr, fc_stds_arr)]

                    fig.add_trace(go.Scatter(
                        x=fc_x + fc_x[::-1],
                        y=fc_upper + fc_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(128, 0, 200, 0.15)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Forecast 95% CI',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc_x, y=fc_means_arr,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='mediumpurple', width=2, dash='dash')
                    ))

                fig.update_layout(
                    title=f"{column}: Observed / Infilled / Forecast",
                    xaxis_title="Time Index",
                    yaxis_title=column,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                MLContainer.plotly_chart(fig, use_container_width=True)

        st.success("Infilled / Forecasted dataset ready.")

        st.download_button(
            "Download Completed Dataset",
            dfFilled.to_csv(index=False),
            file_name="completed_dataset.csv"
        )