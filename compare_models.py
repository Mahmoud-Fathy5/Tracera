"""Compare old vs new model predictions on fat7y.png"""
from inference import VGGGramExtractorV3
import torch, numpy as np, xgboost as xgb, json
from PIL import Image
import torchvision.transforms as transforms

# Load config
with open('model/config_v3.json') as f:
    cfg = json.load(f)

transform = transforms.Compose([
    transforms.Resize((cfg['img_size'], cfg['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(cfg['vgg_mean'], cfg['vgg_std']),
])
img = Image.open('fat7y.png').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# --- OLD model (K=16) ---
ext16 = VGGGramExtractorV3(
    layer_indices=cfg['vgg_layer_indices'],
    channels=cfg['vgg_channels'],
    top_k=16,
)
feats16 = ext16.extract_gram_features(img_tensor)
old_stats = torch.load('model/norm_stats_v3.pt', weights_only=False)
feats16_n = (feats16 - old_stats['feat_mean'].unsqueeze(0)) / (old_stats['feat_std'].unsqueeze(0) + 1e-8)
feats16_n = torch.nan_to_num(feats16_n, nan=0.0, posinf=0.0, neginf=0.0)

with open('model/config_detector_k8_inter_v3.json') as f:
    old_det = json.load(f)
old_idx = old_det.get('feature_indices', None)
if old_idx:
    feats16_det = feats16_n[:, old_idx]
else:
    feats16_det = feats16_n

old_clf = xgb.XGBClassifier()
old_clf.load_model('model/xgb_detector_k8_inter_v3.json')
old_prob = float(old_clf.predict_proba(feats16_det.numpy())[0, 1])
old_verdict = "Fake" if old_prob > 0.5 else "Real"
print(f"OLD model (K=16): fake_prob = {old_prob:.4f}  verdict = {old_verdict}")

# --- NEW model (K=8) ---
ext8 = VGGGramExtractorV3(
    layer_indices=cfg['vgg_layer_indices'],
    channels=cfg['vgg_channels'],
    top_k=8,
)
feats8 = ext8.extract_gram_features(img_tensor)
new_stats = torch.load('model/norm_stats_v3_retrained.pt', weights_only=False)
feats8_n = (feats8 - new_stats['feat_mean'].unsqueeze(0)) / (new_stats['feat_std'].unsqueeze(0) + 1e-8)
feats8_n = torch.nan_to_num(feats8_n, nan=0.0, posinf=0.0, neginf=0.0)

new_clf = xgb.XGBClassifier()
new_clf.load_model('model/xgb_detector_k8_retrained.json')
new_prob = float(new_clf.predict_proba(feats8_n.numpy())[0, 1])
new_verdict = "Fake" if new_prob > 0.5 else "Real"
print(f"NEW model (K=8):  fake_prob = {new_prob:.4f}  verdict = {new_verdict}")

print(f"\nDifference: {old_prob - new_prob:.4f}")
print(f"\n--- Threshold sweep for NEW model ---")
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    v = "Fake" if new_prob > t else "Real"
    print(f"  threshold={t:.2f} -> {v}")
