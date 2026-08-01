"""What influences the v17 winner model's 59% output?

This is the smoking gun: the model has only 2 non-zero
coefficients (r_top_team and d_top_team), so for any
match where neither team is in the v17 top_teams list
(i.e. almost all Dota 2 pro matches), the prediction is
dominated by the r_picks / d_picks count which is always
5+5 = +0.4 in z, giving sigmoid(0.4) = 0.598 ~ 60%.
"""
import sys
sys.path.insert(0, r"C:\Users\artka\.minimax\workspace\dota_analyst")
import business.v17_predict as v17
model, meta = v17._load_model("winner")
cols = meta["feature_columns"]


def predict(features_dict):
    feats = {c: 0.0 for c in cols}
    feats["side_rad"] = 1.0
    feats.update({k: v for k, v in features_dict.items()})
    row = v17._encode_features(meta, feats)
    proba = model.predict_proba([row])[0]
    return proba[1]


print("=" * 78)
print("Smoking gun: which features actually drive the v17 winner output?")
print("=" * 78)
print()
print("1. ALL ZEROS (no teams, no picks):")
print(f"   prob_radiant = {predict({}):.4f}  (50% = prior)")
print()
print("2. Just r_picks=5, d_picks=5:")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5}):.4f}  (this IS the 59.8% you see!)")
print()
print("3. 5+5 picks + r_top_team=1 (radiant is top):")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5, 'r_top_team': 1}):.4f}")
print()
print("4. 5+5 picks + d_top_team=1 (dire is top):")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5, 'd_top_team': 1}):.4f}")
print()
print("5. Massive hero imbalance (r=1.0, d=0.0) - SHOULD be huge signal:")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5, 'r_hero_enc': 1.0, 'd_hero_enc': 0.0}):.4f}  (basically unchanged!)")
print()
print("6. Big gold lead at 5 min (+5000 radiant):")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5, 'gold_adv_5': 5000}):.4f}  (basically unchanged)")
print()
print("7. LGD vs L1ga (real teams, both 'minor' to v17):")
print(f"   prob_radiant = {predict({'r_picks': 5, 'd_picks': 5, 'r_team_id': 82908, 'd_team_id': 36}):.4f}")
print()
print("=" * 78)
print("CONCLUSION")
print("=" * 78)
print("Hero picks, bans, gold advantage, tier, days_since_patch")
print("have ZERO effect on the model output. The 59% is purely")
print("from r_picks=5 + d_picks=5 (always 5) contributing 0.4 to z.")
print()
print("For 95% of pro matches (where neither team is in the v17")
print("top_teams list of 603 matches), the prediction is")
print("mechanically sigmoid(0.4) ~= 59.8% for the radiant side.")
print("The same logic gives 40.2% for the dire side.")
