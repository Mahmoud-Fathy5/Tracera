"""
Tracera — GramNet v3 Inference Module
=====================================
Encapsulates the full deepfake detection pipeline:
  Stage 1: Real vs Fake (XGBoost on Gram eigenvalue features)
  Stage 2: GAN vs Diffusion attribution (if fake)

This module is imported by app.py and initialized once at startup.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as Func
from PIL import Image

# ---------------------------------------------------------------------------
# VGG Gram Matrix Feature Extractor (exact reproduction from training notebook)
# ---------------------------------------------------------------------------

class VGGGramExtractorV3(nn.Module):
    """
    Extracts Compact Spectral Descriptors + 1st-order stats from 3 VGG-16 layers.
    
    Per-layer features:
      - Mean channel activation:       C values
      - Std channel activation:        C values
      - Top-k normalized eigenvalues:  TOP_K_EIGENVALUES values
      - Spectral slope:                1 value
      - Energy bands:                  N_ENERGY_BANDS values
      - Spectral stats:                N_SPECTRAL_STATS values
      Total per layer:                 2*C + TOP_K_EIGENVALUES + 1 + N_ENERGY_BANDS + N_SPECTRAL_STATS
    
    Plus inter-layer features:
      - Cosine similarity of top-k eigenvalue profiles between each layer pair: 3 values

    Total feature dimension ~ 2382
    """

    def __init__(self, layer_indices=None, channels=None, top_k=16, n_energy_bands=4, n_spectral_stats=4):
        super().__init__()
        self.layer_indices = layer_indices or [8, 22, 29]
        self.channels = channels or [128, 512, 512]
        self.top_k = top_k
        self.n_energy_bands = n_energy_bands
        self.n_spectral_stats = n_spectral_stats

        # Load pretrained VGG-16 features (up to relu5_3 = index 29)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        max_idx = max(self.layer_indices) + 1
        self.features = vgg.features[:max_idx]

        # Freeze all VGG parameters — no training needed
        for p in self.features.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def extract_gram_features(self, x):
        """
        Extract Gram matrix eigenvalue features from input image tensor.
        
        Args:
            x: (B, 3, 224, 224) normalized image tensor
        Returns:
            (B, FDIM) feature tensor
        """
        all_layer_features = []
        all_topk_profiles = []  # for inter-layer correlation
        h = x

        for i, layer in enumerate(self.features):
            h = layer(h)

            if i in self.layer_indices:
                B, C, H, W = h.shape
                N = H * W

                # Reshape: (B, C, N)
                F = h.view(B, C, N)

                # -- 1st-Order Features --
                chan_mean = F.mean(dim=2)                     # (B, C)
                chan_std = F.std(dim=2)                       # (B, C)

                # -- 2nd-Order: Gram matrix + eigendecomposition --
                G = torch.bmm(F, F.transpose(1, 2)) / N     # (B, C, C)
                eigvals = torch.linalg.eigvalsh(G)           # (B, C) ascending
                eigvals = eigvals.flip(dims=[1])              # descending
                eigvals = eigvals.clamp(min=0)

                # Normalized eigenvalues
                total_energy = eigvals.sum(dim=1, keepdim=True) + 1e-10
                eigvals_norm = eigvals / total_energy         # (B, C)

                # -- Top-K Eigenvalues --
                k = min(self.top_k, C)
                topk_eigvals = eigvals_norm[:, :k]            # (B, k)
                all_topk_profiles.append(topk_eigvals)

                # -- Spectral Slope --
                log_eigvals = torch.log(eigvals + 1e-10)     # (B, C)
                indices = torch.arange(1, C+1, device=x.device, dtype=torch.float32).unsqueeze(0)  # (1, C)
                x_mean = indices.mean()
                y_mean = log_eigvals.mean(dim=1, keepdim=True)
                cov_xy = ((indices - x_mean) * (log_eigvals - y_mean)).mean(dim=1, keepdim=True)
                var_x = ((indices - x_mean) ** 2).mean()
                slope = cov_xy / (var_x + 1e-10)             # (B, 1)

                # -- Energy Bands --
                q_size = C // self.n_energy_bands
                bands = []
                for q in range(self.n_energy_bands):
                    start = q * q_size
                    end = (q + 1) * q_size if q < self.n_energy_bands - 1 else C
                    band_energy = eigvals_norm[:, start:end].sum(dim=1, keepdim=True)
                    bands.append(band_energy)
                energy_bands = torch.cat(bands, dim=1)        # (B, 4)

                # -- Spectral Statistics (4 per layer) --
                p = eigvals_norm.clamp(min=1e-10)
                entropy = -(p * torch.log(p)).sum(dim=1, keepdim=True)
                eff_rank = torch.exp(entropy)
                cond = torch.log(eigvals[:, 0:1] / (eigvals[:, -1:] + 1e-10) + 1)
                
                eig_mean = eigvals.mean(dim=1, keepdim=True)
                eig_std  = eigvals.std(dim=1, keepdim=True) + 1e-10
                kurtosis = ((eigvals - eig_mean) / eig_std).pow(4).mean(dim=1, keepdim=True) - 3.0

                # Concatenate all features for this layer
                layer_feat = torch.cat([
                    chan_mean,                               # (B, C)
                    chan_std,                                # (B, C)
                    topk_eigvals,                            # (B, k)
                    slope,                                   # (B, 1)
                    energy_bands,                            # (B, 4)
                    entropy, eff_rank, cond, kurtosis,       # (B, 4)
                ], dim=1)

                all_layer_features.append(layer_feat)

        # -- Inter-layer Spectral Correlation --
        inter_layer_feats = []
        for i in range(len(all_topk_profiles)):
            for j in range(i + 1, len(all_topk_profiles)):
                cos_sim = Func.cosine_similarity(
                    all_topk_profiles[i], all_topk_profiles[j], dim=1
                ).unsqueeze(1)
                inter_layer_feats.append(cos_sim)

        all_feat = torch.cat(all_layer_features + inter_layer_feats, dim=1)
        return all_feat


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class GramNetDetector:
    """
    Full inference pipeline: image → features → classification.
    
    Usage:
        detector = GramNetDetector("model/")
        result = detector.predict(image_bytes)
        # result = {"verdict": "Fake", "attribution": "GAN", "confidence": 0.87}
    """

    MAGIC_BYTES = {
        b'\xff\xd8\xff': 'jpeg',
        b'\x89PNG':      'png',
        b'RIFF':         'webp',
    }

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir

        # Load main config
        main_config_path = os.path.join(model_dir, "config_v3.json")
        with open(main_config_path, "r") as f:
            self.config = json.load(f)
            
        # Load detector config
        det_config_path = os.path.join(model_dir, "config_detector_k8_retrained.json")
        with open(det_config_path, "r") as f:
            det_config = json.load(f)
            
        self.feature_indices = det_config.get("feature_indices", None)

        # Load VGG Gram extractor
        self.gram_extractor = VGGGramExtractorV3(
            layer_indices=self.config["vgg_layer_indices"],
            channels=self.config["vgg_channels"],
            top_k=det_config.get("top_k_eigenvalues", 16),
        ).to(self.device)
        self.gram_extractor.eval()

        # Load normalization stats
        stats_path = os.path.join(model_dir, "norm_stats_v3_retrained.pt")
        stats = torch.load(stats_path, map_location=self.device, weights_only=False)
        self.feat_mean = stats["feat_mean"].to(self.device)
        self.feat_std  = stats["feat_std"].to(self.device)
        self.threshold = stats.get("best_thresh", 0.5)

        # Check for manual threshold override via threshold_config.json
        # This allows tuning detection sensitivity without retraining
        threshold_config_path = os.path.join(
            os.path.dirname(model_dir), "threshold_config.json"
        )
        if os.path.exists(threshold_config_path):
            try:
                with open(threshold_config_path, "r") as f:
                    thresh_cfg = json.load(f)
                if thresh_cfg.get("threshold_mode") == "manual":
                    manual_thresh = float(thresh_cfg["threshold"])
                    print(f"[Tracera] Manual threshold override: {self.threshold} → {manual_thresh}")
                    self.threshold = manual_thresh
                # Ensemble weight: how much the new model contributes (0.0–1.0)
                # e.g. 0.55 = new model 55%, old model 45%
                self.ensemble_weight = float(thresh_cfg.get("ensemble_weight", 0.55))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[Tracera] Warning: Could not read threshold_config.json: {e}")
                self.ensemble_weight = 0.55
        else:
            self.ensemble_weight = 0.55

        # Load XGBoost detection classifiers
        import xgboost as xgb

        # Primary: new retrained K=8 detector
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(
            os.path.join(model_dir, "xgb_detector_k8_retrained.json")
        )

        # Ensemble: old K=16 detector as safety net
        # Both models vote and their probabilities are averaged,
        # preserving the retrained model's improvements while
        # catching fakes the new model might be less confident on.
        det_top_k = det_config.get("top_k_eigenvalues", 16)
        old_det_path = os.path.join(model_dir, "xgb_detector_k8_inter_v3.json")
        if det_top_k != 16 and os.path.exists(old_det_path):
            self.old_classifier = xgb.XGBClassifier()
            self.old_classifier.load_model(old_det_path)

            # Load the old K=16 config for feature subsetting
            old_det_config_path = os.path.join(model_dir, "config_detector_k8_inter_v3.json")
            with open(old_det_config_path, "r") as f:
                old_det_config = json.load(f)
            self.old_feature_indices = old_det_config.get("feature_indices", None)

            # K=16 extractor + old norm stats (shared with attribution)
            self.old_extractor = VGGGramExtractorV3(
                layer_indices=self.config["vgg_layer_indices"],
                channels=self.config["vgg_channels"],
                top_k=16,
            ).to(self.device)
            self.old_extractor.eval()
            old_stats_path = os.path.join(model_dir, "norm_stats_v3.pt")
            old_stats = torch.load(old_stats_path, map_location=self.device, weights_only=False)
            self.old_feat_mean = old_stats["feat_mean"].to(self.device)
            self.old_feat_std  = old_stats["feat_std"].to(self.device)
            print(f"[Tracera] Ensemble mode: K=8 (retrained) + K=16 (original)")
        else:
            self.old_classifier = None
            self.old_extractor = None
            self.old_feature_indices = None

        # Load XGBoost attribution classifier
        attr_path = os.path.join(model_dir, "xgb_attribution_v3.json")
        if os.path.exists(attr_path):
            self.attr_classifier = xgb.XGBClassifier()
            self.attr_classifier.load_model(attr_path)
        else:
            self.attr_classifier = None

        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((
                self.config["img_size"],
                self.config["img_size"],
            )),
            transforms.ToTensor(),
            transforms.Normalize(
                self.config["vgg_mean"],
                self.config["vgg_std"],
            ),
        ])

        print(f"[Tracera] Model loaded from {model_dir}")
        print(f"[Tracera] Feature dim: {self.config.get('fdim', stats.get('fdim', 'unknown'))}")
        print(f"[Tracera] Detection threshold: {self.threshold}")
        print(f"[Tracera] Attribution model: {'loaded' if self.attr_classifier else 'not found'}")
        print(f"[Tracera] Device: {self.device}")

    @staticmethod
    def validate_image_bytes(file_bytes: bytes) -> bool:
        """
        Validates that file_bytes begin with a known image magic signature.
        """
        for magic, _ in GramNetDetector.MAGIC_BYTES.items():
            if file_bytes[:len(magic)] == magic:
                return True
        return False

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """
        Run full inference pipeline on raw image bytes.
        """
        import io

        # Open and preprocess image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # ---- New model (K=8 retrained) ----
        features = self.gram_extractor.extract_gram_features(img_tensor)
        features = (features - self.feat_mean.unsqueeze(0)) / (
            self.feat_std.unsqueeze(0) + 1e-8
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_indices is not None:
            det_features = features[:, self.feature_indices]
        else:
            det_features = features

        new_prob = float(self.classifier.predict_proba(det_features.cpu().numpy())[0, 1])

        # ---- Old model (K=16 original) for ensemble ----
        if self.old_classifier is not None:
            old_features = self.old_extractor.extract_gram_features(img_tensor)
            old_features = (old_features - self.old_feat_mean.unsqueeze(0)) / (
                self.old_feat_std.unsqueeze(0) + 1e-8
            )
            old_features = torch.nan_to_num(old_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Keep full K=16 features for attribution
            old_feats_full_np = old_features.cpu().numpy()

            if self.old_feature_indices is not None:
                old_det_features = old_features[:, self.old_feature_indices]
            else:
                old_det_features = old_features

            old_prob = float(self.old_classifier.predict_proba(old_det_features.cpu().numpy())[0, 1])

            # Ensemble: weighted average of both model probabilities
            w = self.ensemble_weight  # new model weight
            fake_prob = w * new_prob + (1 - w) * old_prob
        else:
            fake_prob = new_prob
            old_feats_full_np = None

        verdict = "Fake" if fake_prob > self.threshold else "Real"

        # Attribution (only if fake)
        attribution = None
        attribution_confidence = None
        if verdict == "Fake" and self.attr_classifier is not None:
            # Use K=16 features for attribution (it was trained on 2382-dim)
            if old_feats_full_np is not None:
                attr_feats_np = old_feats_full_np
            else:
                attr_feats_np = features.cpu().numpy()

            attr_prob = float(self.attr_classifier.predict_proba(attr_feats_np)[0, 1])
            attribution = "Diffusion" if attr_prob > 0.5 else "GAN"
            attribution_confidence = round(
                attr_prob if attribution == "Diffusion" else (1 - attr_prob), 4
            )

        # Calculate UX-friendly confidence scaled from 50% to 100%
        if verdict == "Fake":
            margin = max(0.0, fake_prob - self.threshold) / max(1e-5, 1.0 - self.threshold)
        else:
            margin = max(0.0, self.threshold - fake_prob) / max(1e-5, self.threshold)
            
        # Apply a square root to boost small margins so they don't look like coin flips
        ux_confidence = 0.5 + 0.5 * (margin ** 0.5)

        return {
            "verdict": verdict,
            "confidence": round(ux_confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "attribution": attribution,
            "attribution_confidence": attribution_confidence,
        }
