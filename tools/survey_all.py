"""Load every stock + loose track; record pass/fail."""
import os, subprocess, sys, time

EXE = r"C:\Users\joshu\Downloads\evoview\target\release\evoview.exe"
PKG = r"C:\Program Files (x86)\Steam\steamapps\common\Assetto Corsa EVO\content.kspkg"
LOOSE = r"C:\Users\joshu\Saved Games\ACE\mods\content\tracks"
OUT = r"C:\Users\joshu\Downloads\acecm\data\survey"
os.makedirs(OUT, exist_ok=True)

def list_stock():
    p = subprocess.run([EXE, PKG], capture_output=True, text=True, timeout=120)
    ids = []
    for line in (p.stdout or "").splitlines():
        if line.strip().startswith("track"):
            ids.append(line.split()[-1])
    return ids

def run_one(name, cmd):
    shot = os.path.join(OUT, name.replace("\\", "_") + ".png")
    full = cmd + ["--shot", shot, "--size", "960x540"]
    t0 = time.time()
    try:
        p = subprocess.run(full, capture_output=True, text=True, timeout=180)
        dt = time.time() - t0
        tail = ((p.stdout or "") + "\n" + (p.stderr or "")).strip().splitlines()
        summary = " | ".join(tail[-6:])
        ok = p.returncode == 0 and os.path.isfile(shot)
        print(f"{'OK' if ok else 'FAIL':4} {dt:6.1f}s {name}  {summary[:180]}")
        return ok, dt, summary
    except subprocess.TimeoutExpired:
        print(f"FAIL {180:6.1f}s {name}  TIMEOUT")
        return False, 180, "TIMEOUT"
    except Exception as ex:
        print(f"FAIL    - {name}  {ex}")
        return False, 0, str(ex)

def main():
    stock = list_stock()
    print(f"stock tracks: {len(stock)}")
    loose = []
    if os.path.isdir(LOOSE):
        for n in sorted(os.listdir(LOOSE)):
            p = os.path.join(LOOSE, n)
            if os.path.isdir(p) and (
                os.path.isdir(os.path.join(p, "static_mesh"))
                or os.path.isdir(os.path.join(p, "meshes"))
                or os.path.isdir(os.path.join(p, "content"))
            ):
                loose.append(n)
    print(f"loose tracks: {len(loose)}")
    results = []
    for tid in stock:
        ok, dt, msg = run_one(tid, [EXE, PKG, "--track", tid])
        results.append((ok, "stock", tid, dt, msg))
    for tid in loose:
        folder = os.path.join(LOOSE, tid)
        ok, dt, msg = run_one("loose_" + tid, [EXE, folder, "--track", "--base", PKG])
        results.append((ok, "loose", tid, dt, msg))
    print("\n==== SUMMARY")
    fails = [r for r in results if not r[0]]
    print(f"passed {len(results)-len(fails)}/{len(results)}")
    for r in fails:
        print(f"  FAIL {r[1]} {r[2]}: {r[4][:160]}")

if __name__ == "__main__":
    main()
