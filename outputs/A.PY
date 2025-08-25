import json, numpy as np, torch
print("labels.json:", json.load(open("labels.json")))
ths = np.load("thresholds.npy"); print("thresholds:", np.round(ths,3))
sd = torch.load("final_model.pt", map_location="cpu")
sd = sd.get("state_dict", sd)
print("attention.shape:", tuple(sd["attention"].shape))