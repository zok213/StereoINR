import os, json

base_dir = r"d:\Gitrepo\Active Stereo RL"
targets = [
    os.path.join(base_dir, "phase1_reimplement", "StereoINR_official", "StereoINR_DualT4_Full_Reimplementation.ipynb"),
    os.path.join(base_dir, "phase1_reimplement", "StereoINR_official", "notebooks", "StereoINR_DualT4_Full_Reimplementation.ipynb"),
    os.path.join(base_dir, "StereoINR_DualT4_Full_Reimplementation.ipynb"),
    r"C:\Users\Admin\Downloads\stereoinr.ipynb"
]

for p in targets:
    if not os.path.exists(p): continue
    nb = json.load(open(p, 'r', encoding='utf-8'))
    cell19_code = "".join(nb['cells'][19]['source'])
    cell19_code = cell19_code.replace("(19.74, 0.6630)", "(19.46, 0.7027)")
    cell19_code = cell19_code.replace("'StereoINR (Live)'", "'StereoINR (Ours)'")
    cell19_code = cell19_code.replace('"StereoINR (Live)"', '"StereoINR (Ours)"')
    nb['cells'][19]['source'] = [l + "\n" for l in cell19_code.splitlines()]
    if nb['cells'][19]['source']:
        nb['cells'][19]['source'][-1] = nb['cells'][19]['source'][-1].rstrip("\n")
    json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1)
print("Cleaned Cell 19 source code successfully!")
