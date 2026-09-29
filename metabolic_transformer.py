"""
MetabolicTransformer: Multi-Modal Transformer for Metabolic Vulnerability Discovery

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_recall_fscore_support, average_precision_score
)
from scipy.stats import spearmanr
import warnings
import gc
import os

warnings.filterwarnings('ignore')


#  FEATURE TOKENIZATION LAYER

class FeatureTokenizer(nn.Module):
    """
    Converts each scalar feature into a d-dimensional token embedding.
    
    For feature i with value x_i:
        token_i = x_i * W_i + b_i
    
    Each feature gets its own learned linear projection, allowing the model
    to learn feature-specific representations.
    """
    
    def __init__(self, n_features, d_token):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        # Each feature gets its own projection weights
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        
        # Initialisation (Xavier uniform)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_features)
        Returns:
            tokens: (batch_size, n_features, d_token)
        """
        # x[:, :, None] -> (batch, n_features, 1)
        # self.weight -> (n_features, d_token)
        # Result -> (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight[None] + self.bias[None]


#  MULTI-HEAD SELF-ATTENTION

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention with optional attention output.
    
    This is where the model learns which pathway-flux interactions matter:
    - Pathway tokens can attend to flux tokens (cross-modal)
    - Flux tokens can attend to pathway tokens
    - Same-modality interactions are also captured
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            return_attention: if True, also return attention weights
        Returns:
            output: (batch_size, seq_len, d_model)
            attention: (batch_size, n_heads, seq_len, seq_len) [optional]
        """
        B, S, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn  # Save before dropout
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        out = self.W_o(out)
        
        if return_attention:
            return out, attn_weights
        return out


# TRANSFORMER ENCODER BLOCK

class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block with pre-norm architecture.
    
    Pre-norm (LayerNorm before attention/FFN) is more stable for training
    and doesn't require learning rate warmup.
    """
    
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, return_attention=False):
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(normed, return_attention=True)
        else:
            attn_out = self.attn(normed)
        x = x + attn_out
        
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


#  METABOLIC TRANSFORMER

class MetabolicTransformer(nn.Module):
    """
    Feature Tokenization Transformer for multi-modal metabolic data.
    
    Concatenates pathway activity scores (expression-derived) with
    metabolic flux estimates (scFEA-derived) into a unified transformer.
    
    Architecture:
        Input features → Feature Tokenizer → [CLS] + Token Sequence
        → Modality Embeddings → Transformer Encoder Blocks
        → [CLS] representation → Classification Head
    
    Args:
        n_expression_features: Number of pathway activity features
        n_flux_features: Number of scFEA flux module features
        n_covariate_features: Number of technical covariates
        d_token: Dimension of each feature token
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ffn_factor: FFN hidden dimension = d_token * d_ffn_factor
        dropout: Dropout rate
    """
    
    VERSION = "2.0.0"
    
    def __init__(
        self,
        n_expression_features=20,
        n_flux_features=76,
        n_covariate_features=3,
        d_token=64,
        n_heads=4,
        n_layers=3,
        d_ffn_factor=2,
        dropout=0.2
    ):
        super().__init__()
        
        self.n_expression = n_expression_features
        self.n_flux = n_flux_features
        self.n_covariates = n_covariate_features
        self.n_total = n_expression_features + n_flux_features + n_covariate_features
        self.d_token = d_token
        
        # Feature tokenizer: each feature → d_token dimensional embedding
        self.tokenizer = FeatureTokenizer(self.n_total, d_token)
        
        # [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Modality embeddings: 0=expression, 1=flux, 2=covariate
        self.modality_embedding = nn.Embedding(3, d_token)
        
        # Transformer encoder
        d_ffn = d_token * d_ffn_factor
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_token)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )
    
    def _get_modality_ids(self, batch_size, device):
        """Create modality ID tensor for each feature position."""
        ids = torch.cat([
            torch.zeros(self.n_expression, dtype=torch.long),   # Expression
            torch.ones(self.n_flux, dtype=torch.long),          # Flux
            torch.full((self.n_covariates,), 2, dtype=torch.long)  # Covariate
        ]).to(device)
        return ids.unsqueeze(0).expand(batch_size, -1)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, n_total_features)
               Features ordered as: [expression | flux | covariates]
            return_attention: if True, return attention weights from all layers
        
        Returns:
            logits: (batch_size, 1) — raw logits (apply sigmoid for probability)
            attentions: list of (batch_size, n_heads, seq_len, seq_len) [optional]
        """
        B = x.shape[0]
        device = x.device
        
        # Tokenize features → (B, n_features, d_token)
        tokens = self.tokenizer(x)
        
        #  Add modality embeddings
        modality_ids = self._get_modality_ids(B, device)
        tokens = tokens + self.modality_embedding(modality_ids)
        
        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+n_features, d_token)
        
        #  Transformer encoder
        attentions = []
        for block in self.blocks:
            if return_attention:
                tokens, attn = block(tokens, return_attention=True)
                attentions.append(attn)
            else:
                tokens = block(tokens)
        
        # Extract [CLS] representation
        cls_repr = self.final_norm(tokens[:, 0])
        
        # Classification
        logits = self.classifier(cls_repr)
        
        if return_attention:
            return logits, attentions
        return logits
    
    def get_attention_weights(self, x):
        """Extract attention weights for interpretability analysis."""
        self.eval()
        with torch.no_grad():
            _, attentions = self.forward(x, return_attention=True)
        return attentions
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# TRAINING UTILITIES

class EarlyStopping:
    """Early stopping with patience and best model saving."""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits.squeeze(), y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss + predictions."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits.squeeze(), y_batch)
            
            probs = torch.sigmoid(logits.squeeze())
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
            total_loss += loss.item()
            n_batches += 1
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    return total_loss / n_batches, preds, labels


# FEATURE PREPARATION

def prepare_multimodal_features(
    adata,
    pathway_cols,
    flux_df,
    covariate_cols=None,
    flux_top_n=None
):
    """
    Concatenate expression pathway scores with scFEA flux features.
    
    Args:
        adata: AnnData object with pathway scores in .obs
        pathway_cols: List of pathway column names in adata.obs
        flux_df: DataFrame with flux values (spots × modules)
        covariate_cols: List of technical covariate columns
        flux_top_n: If set, use only top N most variable flux modules
    
    Returns:
        X_combined: Combined feature matrix
        feature_names: List of all feature names
        feature_types: List of feature type labels ('expression', 'flux', 'covariate')
    """
    if covariate_cols is None:
        covariate_cols = ['log1p_total_counts', 'pct_counts_mt', 'pct_counts_ribo']
    
    print("PREPARING MULTI-MODAL FEATURES")
    
    #  Expression pathway scores
    X_expr = adata.obs[pathway_cols].values
    print(f"  Expression features: {X_expr.shape[1]} pathways")
    
    #  Flux features
    # Align flux data with adata spots
    common_spots = adata.obs_names.intersection(flux_df.index)
    if len(common_spots) < len(adata.obs_names):
        print(f"  Warning: {len(adata.obs_names) - len(common_spots)} spots without flux data")
        # Fill missing with 0
        flux_aligned = pd.DataFrame(0.0, index=adata.obs_names, columns=flux_df.columns)
        flux_aligned.loc[common_spots] = flux_df.loc[common_spots]
    else:
        flux_aligned = flux_df.loc[adata.obs_names]
    
    # Optional: select top N most variable flux modules
    if flux_top_n is not None:
        flux_var = flux_aligned.var().sort_values(ascending=False)
        top_modules = flux_var.index[:flux_top_n].tolist()
        flux_aligned = flux_aligned[top_modules]
        print(f"  Flux features: {len(top_modules)} modules (top {flux_top_n} by variance)")
    else:
        print(f"  Flux features: {flux_aligned.shape[1]} modules")
    
    X_flux = flux_aligned.values
    flux_cols = flux_aligned.columns.tolist()
    
    #  Technical covariates
    available_covs = [c for c in covariate_cols if c in adata.obs.columns]
    X_cov = adata.obs[available_covs].values
    print(f"  Covariate features: {len(available_covs)}")
    
    # Concatenate: [expression | flux | covariates]
    X_combined = np.hstack([X_expr, X_flux, X_cov])
    
    feature_names = pathway_cols + flux_cols + available_covs
    feature_types = (
        ['expression'] * len(pathway_cols) + 
        ['flux'] * len(flux_cols) + 
        ['covariate'] * len(available_covs)
    )
    
    print(f"\n  TOTAL: {X_combined.shape[1]} features")
    print(f"    Expression: {len(pathway_cols)}")
    print(f"    Flux:       {len(flux_cols)}")
    print(f"    Covariates: {len(available_covs)}")
    
    return X_combined, feature_names, feature_types


#  SPATIAL CROSS-VALIDATION PIPELINE

def spatial_cv_transformer(
    X, y, spatial_blocks, feature_names, feature_types,
    n_splits=5,
    d_token=64,
    n_heads=4,
    n_layers=3,
    dropout=0.2,
    lr=1e-4,
    weight_decay=1e-2,
    epochs=150,
    batch_size=256,
    patience=20,
    device=None,
    verbose=True
):
    """
    Train MetabolicTransformer with spatial block cross-validation.
    
    Returns:
        results: dict with per-fold metrics and aggregated results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_expr = sum(1 for t in feature_types if t == 'expression')
    n_flux = sum(1 for t in feature_types if t == 'flux')
    n_cov = sum(1 for t in feature_types if t == 'covariate')
    
    print(f"\nMETABOLIC TRANSFORMER — SPATIAL CROSS-VALIDATION")
    print(f"{'='*55}")
    print(f"  Features: {len(feature_names)} total")
    print(f"    Expression: {n_expr} | Flux: {n_flux} | Covariates: {n_cov}")
    print(f"  Architecture: d={d_token}, heads={n_heads}, layers={n_layers}")
    print(f"  Training: lr={lr}, wd={weight_decay}, epochs={epochs}")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")
    
    cv = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    all_preds = np.zeros(len(y))
    all_attentions = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=spatial_blocks)):
        if verbose:
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardise (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        # Create model
        model = MetabolicTransformer(
            n_expression_features=n_expr,
            n_flux_features=n_flux,
            n_covariate_features=n_cov,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)
        
        if fold == 0 and verbose:
            print(f"  Parameters: {model.count_parameters():,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        early_stop = EarlyStopping(patience=patience)
        
        # Training loop
        best_auc = 0
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except ValueError:
                val_auc = 0.5
            
            early_stop(val_auc, model)
            
            if val_auc > best_auc:
                best_auc = val_auc
            
            if early_stop.early_stop:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1} | Best AUC: {best_auc:.4f}")
                break
        
        # Load best model and get final predictions
        early_stop.load_best(model)
        _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        
        # Compute metrics
        fold_auc = roc_auc_score(test_labels, test_preds)
        fold_acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
        fold_f1 = f1_score(test_labels, (test_preds > 0.5).astype(int))
        fold_prauc = average_precision_score(test_labels, test_preds)
        
        all_preds[test_idx] = test_preds
        
        # Extract attention weights for interpretability
        model.eval()
        with torch.no_grad():
            _, attn = model(X_test_t[:min(500, len(X_test_t))].to(device), return_attention=True)
            # Average over samples and heads for last layer
            avg_attn = attn[-1].mean(dim=(0, 1)).cpu().numpy()
            all_attentions.append(avg_attn)
        
        fold_results.append({
            'fold': fold + 1,
            'auc': fold_auc,
            'accuracy': fold_acc,
            'f1': fold_f1,
            'pr_auc': fold_prauc,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        if verbose:
            print(f"  Fold {fold+1}: AUC={fold_auc:.4f} | Acc={fold_acc:.4f} | F1={fold_f1:.4f}")
        
        # Cleanup
        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate results
    fold_df = pd.DataFrame(fold_results)
    
    print(f"\n{'='*55}")
    print(f"RESULTS SUMMARY")
    print(f"  AUC:    {fold_df['auc'].mean():.4f} +/- {fold_df['auc'].std():.4f}")
    print(f"  Acc:    {fold_df['accuracy'].mean():.4f} +/- {fold_df['accuracy'].std():.4f}")
    print(f"  F1:     {fold_df['f1'].mean():.4f} +/- {fold_df['f1'].std():.4f}")
    print(f"  PR-AUC: {fold_df['pr_auc'].mean():.4f} +/- {fold_df['pr_auc'].std():.4f}")
    print(f"{'='*55}")
    
    return {
        'fold_results': fold_df,
        'all_preds': all_preds,
        'attentions': all_attentions,
        'feature_names': feature_names,
        'feature_types': feature_types
    }


#  ATTENTION ANALYSIS FOR INTERPRETABILITY

def analyse_attention(
    results,
    top_k=15,
    save_dir='results/transformer'
):
    """
    Analyse attention weights to identify key feature interactions.
    
    Extracts:
    1. Feature importance from [CLS] attention
    2. Cross-modal attention patterns (expression ↔ flux)
    3. Top pathway-flux interaction pairs
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(save_dir, exist_ok=True)
    
    feature_names = results['feature_names']
    feature_types = results['feature_types']
    attentions = results['attentions']
    
    # Average attention across folds
    avg_attention = np.mean(attentions, axis=0)
    
    # Feature names with [CLS] prepended
    token_names = ['[CLS]'] + feature_names
    token_types = ['cls'] + feature_types
    
    print("\nATTENTION ANALYSIS")
    
    
    # [CLS] attention → Feature importance
    cls_attention = avg_attention[0, 1:]  
    
    # Normalise
    cls_attention = cls_attention / cls_attention.sum()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'type': feature_types,
        'attention': cls_attention
    }).sort_values('attention', ascending=False)
    
    print("\nTop Features by [CLS] Attention:")
    for _, row in importance_df.head(top_k).iterrows():
        marker = "E" if row['type'] == 'expression' else ("F" if row['type'] == 'flux' else "C")
        print(f"  [{marker}] {row['feature']}: {row['attention']:.4f}")
    
    #  Cross-modal attention
    expr_idx = [i+1 for i, t in enumerate(feature_types) if t == 'expression']
    flux_idx = [i+1 for i, t in enumerate(feature_types) if t == 'flux']
    
    # Expression → Flux attention
    if expr_idx and flux_idx:
        expr_to_flux = avg_attention[np.ix_(expr_idx, flux_idx)]
        flux_to_expr = avg_attention[np.ix_(flux_idx, expr_idx)]
        
        expr_names = [feature_names[i-1] for i in expr_idx]
        flux_names = [feature_names[i-1] for i in flux_idx]
        
        # Find top interactions
        interactions = []
        for i, e_name in enumerate(expr_names):
            for j, f_name in enumerate(flux_names):
                score = (expr_to_flux[i, j] + flux_to_expr[j, i]) / 2
                interactions.append({
                    'pathway': e_name,
                    'flux_module': f_name,
                    'attention_score': score
                })
        
        interaction_df = pd.DataFrame(interactions).sort_values('attention_score', ascending=False)
        
        print(f"\nTop Cross-Modal Interactions (Expression <-> Flux):")
        for _, row in interaction_df.head(10).iterrows():
            print(f"  {row['pathway']} <-> {row['flux_module']}: {row['attention_score']:.4f}")
        
        interaction_df.to_csv(f'{save_dir}/cross_modal_interactions.csv', index=False)
    
    #  Visualisation: Feature importance by modality
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of top features
    top_features = importance_df.head(top_k)
    colors = ['#0D5C63' if t == 'expression' else '#E85A4F' if t == 'flux' else '#888888' 
              for t in top_features['type']]
    
    axes[0].barh(range(len(top_features)), top_features['attention'].values, color=colors)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features['feature'].values, fontsize=9)
    axes[0].set_xlabel('Attention Weight')
    axes[0].set_title('Feature Importance ([CLS] Attention)')
    axes[0].invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#0D5C63', label='Expression'),
        Patch(facecolor='#E85A4F', label='Flux'),
        Patch(facecolor='#888888', label='Covariate')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # Modality-level attention distribution
    modality_attention = importance_df.groupby('type')['attention'].sum()
    colors_pie = {'expression': '#0D5C63', 'flux': '#E85A4F', 'covariate': '#888888'}
    axes[1].pie(
        modality_attention.values,
        labels=[f"{k.title()}\n({v:.1%})" for k, v in modality_attention.items()],
        colors=[colors_pie.get(k, '#888888') for k in modality_attention.index],
        autopct='',
        startangle=90
    )
    axes[1].set_title('Attention by Modality')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {save_dir}/attention_analysis.png")
    
    # Cross-modal attention heatmap 
    if len(expr_names) <= 25 and len(flux_names) <= 40:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            expr_to_flux,
            xticklabels=flux_names,
            yticklabels=expr_names,
            cmap='YlOrRd',
            ax=ax
        )
        ax.set_title('Cross-Modal Attention: Expression → Flux')
        ax.set_xlabel('Flux Modules')
        ax.set_ylabel('Pathways')
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/cross_modal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {save_dir}/cross_modal_heatmap.png")
    
    # Save results
    importance_df.to_csv(f'{save_dir}/feature_importance.csv', index=False)
    print(f"  Saved: {save_dir}/feature_importance.csv")
    
    return importance_df, interaction_df if expr_idx and flux_idx else None


# LOPO VALIDATION WITH TRANSFORMER

def lopo_transformer(
    patient_data_dict,
    pathway_cols,
    flux_dir,
    covariate_cols=None,
    flux_top_n=None,
    d_token=64,
    n_heads=4,
    n_layers=3,
    dropout=0.2,
    lr=1e-4,
    epochs=100,
    batch_size=256,
    patience=20,
    device=None,
    verbose=True
):
    """
    Leave-One-Patient-Out validation with MetabolicTransformer.
    
    Args:
        patient_data_dict: {patient_id: {'adata': adata, 'flux': flux_df}}
        pathway_cols: Shared pathway column names
        flux_dir: Directory with flux results
        ... (training hyperparameters)
    
    Returns:
        lopo_results: DataFrame with per-patient metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if covariate_cols is None:
        covariate_cols = ['log1p_total_counts', 'pct_counts_mt', 'pct_counts_ribo']
    
    patient_ids = sorted(patient_data_dict.keys())
    
    print(f"\nLOPO VALIDATION — METABOLIC TRANSFORMER")
    print(f"{'='*55}")
    print(f"  Patients: {patient_ids}")
    print(f"  Device: {device}")
    print(f"{'='*55}\n")
    
    lopo_results = []
    
    for test_patient in patient_ids:
        train_patients = [p for p in patient_ids if p != test_patient]
        
        print(f"\n--- Test: Patient {test_patient} | Train: {train_patients} ---")
        
        # Collect training data
        X_train_list, y_train_list = [], []
        for p in train_patients:
            data = patient_data_dict[p]
            X_p, fnames, ftypes = prepare_multimodal_features(
                data['adata'], pathway_cols, data['flux'],
                covariate_cols, flux_top_n
            )
            y_p = data['adata'].obs['target_proliferation_binary'].values
            X_train_list.append(X_p)
            y_train_list.append(y_p)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Test data
        test_data = patient_data_dict[test_patient]
        X_test, _, _ = prepare_multimodal_features(
            test_data['adata'], pathway_cols, test_data['flux'],
            covariate_cols, flux_top_n
        )
        y_test = test_data['adata'].obs['target_proliferation_binary'].values
        
        # Feature counts
        n_expr = len(pathway_cols)
        n_flux = X_train.shape[1] - n_expr - len(covariate_cols)
        n_cov = len(covariate_cols)
        
        # Standardise
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        
        # Tensors
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        # Model
        model = MetabolicTransformer(
            n_expression_features=n_expr,
            n_flux_features=n_flux,
            n_covariate_features=n_cov,
            d_token=d_token,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay if 'weight_decay' in dir() else 1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        early_stop = EarlyStopping(patience=patience)
        
        # Train
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
            except ValueError:
                val_auc = 0.5
            
            early_stop(val_auc, model)
            if early_stop.early_stop:
                break
        
        # Best model evaluation
        early_stop.load_best(model)
        _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        
        auc = roc_auc_score(test_labels, test_preds)
        acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
        prec, rec, f1, _ = precision_recall_fscore_support(
            test_labels, (test_preds > 0.5).astype(int), average='binary'
        )
        
        lopo_results.append({
            'patient': test_patient,
            'auc': auc,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'n_spots': len(test_labels)
        })
        
        print(f"  Patient {test_patient}: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
        
        del model, optimizer
        gc.collect()
    
    lopo_df = pd.DataFrame(lopo_results)
    
    print(f"\n{'='*55}")
    print(f"LOPO SUMMARY — METABOLIC TRANSFORMER")
    print(f"  Mean AUC: {lopo_df['auc'].mean():.4f} +/- {lopo_df['auc'].std():.4f}")
    print(f"  Mean Acc: {lopo_df['accuracy'].mean():.4f} +/- {lopo_df['accuracy'].std():.4f}")
    print(f"  Mean F1:  {lopo_df['f1'].mean():.4f} +/- {lopo_df['f1'].std():.4f}")
    print(f"{'='*55}")
    
    return lopo_df


# MODEL COMPARISON UTILITY

def compare_models(baseline_results, transformer_results, model_names=None):
    """
    Statistical comparison between MetabolicVulnerabilityNet and MetabolicTransformer.
    """
    from scipy.stats import wilcoxon, ttest_rel
    
    if model_names is None:
        model_names = ['MetabolicVulnerabilityNet', 'MetabolicTransformer']
    
    print(f"\nMODEL COMPARISON")
  
    
    metrics = ['auc', 'accuracy', 'f1']
    
    for metric in metrics:
        base_vals = baseline_results[metric].values
        trans_vals = transformer_results[metric].values
        
        base_mean = base_vals.mean()
        trans_mean = trans_vals.mean()
        improvement = trans_mean - base_mean
        
        try:
            stat, p_val = wilcoxon(trans_vals, base_vals)
        except ValueError:
            p_val = 1.0
        
        sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"\n  {metric.upper()}:")
        print(f"    {model_names[0]}: {base_mean:.4f} +/- {base_vals.std():.4f}")
        print(f"    {model_names[1]}: {trans_mean:.4f} +/- {trans_vals.std():.4f}")
        print(f"    Improvement: {improvement:+.4f} (p={p_val:.4f}) {sig}")
    
 



