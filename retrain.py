import os
import glob
import torch
import numpy as np
import xgboost as xgb
import json
from inference import GramNetDetector

# ==============================================================================
# TRACERA - INCREMENTAL RETRAINING SCRIPT
# ==============================================================================
# CAN YOU RETRAIN WITH JUST 1 OR 2 IMAGES WITHOUT PRIOR DATA?
# Technically: YES. XGBoost supports incremental training by passing the
# existing model to the `xgb_model` parameter in `fit()`.
#
# Practically: IT IS HIGHLY DISCOURAGED without original data.
# Tree-based models (like XGBoost) don't learn smoothly from single instances
# like neural networks with low learning rates. If you incrementally add trees
# for just 1 or 2 images, the model will drastically overfit to the new data
# and suffer from catastrophic forgetting, ruining the 88.58% global accuracy.
#
# RECOMMENDATION:
# To safely retrain, you MUST have at least a representative subset of the
# original training dataset. Combine your new images with the old dataset,
# then fit the XGBoost model on the combined data.
# ==============================================================================

def extract_features_from_dir(directory, detector):
    """
    Extracts 2358-dimensional spectral features for all images in a directory.
    """
    features_list = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found.")
        return features_list

    # Support multiple extensions
    file_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        file_paths.extend(glob.glob(os.path.join(directory, ext)))
        
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            
            if not detector.validate_image_bytes(img_bytes):
                continue
                
            # We can lightly modify predict pipeline to just extract features
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_tensor = detector.transform(img).unsqueeze(0).to(detector.device)
            
            with torch.no_grad():
                feat = detector.gram_extractor.extract_gram_features(img_tensor)
                
                # Z-score normalization
                feat = (feat - detector.feat_mean.unsqueeze(0)) / (detector.feat_std.unsqueeze(0) + 1e-8)
                feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Subset to 2358 features for the K=8 model
                if detector.feature_indices is not None:
                    feat = feat[:, detector.feature_indices]
                    
            features_list.append(feat.cpu().numpy()[0])
            print(f"Extracted features for {path}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return features_list

def retrain_incremental(real_dir, fake_dir, old_model_path, new_model_path):
    print("Loading existing inference pipeline...")
    detector = GramNetDetector("model/")
    
    print("\n--- Extracting Features for Real Images ---")
    real_features = extract_features_from_dir(real_dir, detector)
    
    print("\n--- Extracting Features for Fake Images ---")
    fake_features = extract_features_from_dir(fake_dir, detector)
    
    X_new = []
    y_new = []
    
    if real_features:
        X_new.extend(real_features)
        y_new.extend([0] * len(real_features))
        
    if fake_features:
        X_new.extend(fake_features)
        y_new.extend([1] * len(fake_features))
        
    if not X_new:
        print("\nNo new images found to train on. Exiting.")
        return
        
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    
    print(f"\nTraining on {len(X_new)} new samples ({len(real_features)} Real, {len(fake_features)} Fake)...")
    
    # Initialize XGBoost Classifier with Incremental Learning parameters
    # The new trees added will attempt to correct errors from the old model
    rf = xgb.XGBClassifier()
    
    # Note: Incremental training requires the exact original objective and parameters
    rf.fit(X_new, y_new, xgb_model=old_model_path)
    
    rf.save_model(new_model_path)
    print(f"\nRetrained model saved successfully to: {new_model_path}")
    print("Don't forget to update inference.py if you want to use the retrained model instead of the default one.")

if __name__ == "__main__":
    # Example usage:
    # 1. Place some real images in data/new_real/
    # 2. Place some fake images in data/new_fake/
    # 3. Run this script!
    
    OLD_MODEL = "model/xgb_detector_k8_inter_v3.json"
    NEW_MODEL = "model/xgb_detector_k8_inter_v3_retrained.json"
    
    print("=" * 60)
    print("WARNING: Incremental training with a very small batch of data")
    print("will severely overfit and destroy the global 88.58% accuracy.")
    print("Please read the instructions inside the code for context.")
    print("=" * 60)
    
    # retrain_incremental(
    #     real_dir="data/new_real", 
    #     fake_dir="data/new_fake", 
    #     old_model_path=OLD_MODEL, 
    #     new_model_path=NEW_MODEL
    # )
