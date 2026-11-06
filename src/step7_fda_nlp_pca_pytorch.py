import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("data/processed/mart_careledger_claims_360.csv")

print("1. Vectorizing OpenFDA Adverse Event Narratives & Compressing with PCA...")
tfidf_mat = TfidfVectorizer(max_features=48).fit_transform(df["fda_clinical_narrative"].fillna("ROUTINE")).toarray()
pca = PCA(n_components=4, random_state=42)
pca_comps = pca.fit_transform(tfidf_mat)
print(f"   -> Top 4 Principal Components Explained Variance = {pca.explained_variance_ratio_.sum()*100:.2f}%")

for i in range(4):
    df[f"fda_pca_{i+1}"] = pca_comps[:, i]

cols = [
    "charge_to_allowed_markup_ratio", "denial_appeals_count", "provider_quality_star_score",
    "prior_auth_turnaround_hours", "quarterly_submitted_usd", "quarterly_allowed_usd",
    "fda_pca_1", "fda_pca_2", "fda_pca_3", "fda_pca_4"
]

X = StandardScaler().fit_transform(df[cols].values).astype(np.float32)
y = df["claim_audit_flagged_60d"].values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"2. Training PyTorch Clinical & Claims Neural Network on MacBook Device: {device}")

class CareLedgerNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = CareLedgerNet(X_train.shape[1]).to(device)
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1.0, y_train.sum())], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=512, shuffle=True)

model.train()
for epoch in range(4):
    loss_sum = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"   -> Epoch {epoch+1}/4 | BCE Loss = {loss_sum/len(loader):.4f}")

model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(torch.tensor(X_test).to(device))).cpu().numpy()

print(f"\nSTEP 9 VERIFIED! PyTorch + OpenFDA PCA Test ROC-AUC = {roc_auc_score(y_test, preds):.4f} | PR-AUC = {average_precision_score(y_test, preds):.4f}")