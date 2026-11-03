import os
root = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
for track in sorted(os.listdir(root)):
    md = os.path.join(root, track, "static_mesh")
    if not os.path.isdir(md):
        continue
    hits = []
    for fn in os.listdir(md):
        low = fn.lower()
        if any(x in low for x in ("dummy", "shadow", "lowres", "physics", "collider", "decal")):
            hits.append(fn)
    if hits:
        print(track, len(hits), hits[:12])
