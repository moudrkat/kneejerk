#!/usr/bin/env python3
"""Images: the same picture and questions through systemone-cli and through llama-server, probabilities compared."""
import argparse, base64, json, subprocess, sys, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--cli"); ap.add_argument("--model"); ap.add_argument("--mmproj"); ap.add_argument("--image"); ap.add_argument("--server")
ap.add_argument("--threads", default="8")
a = ap.parse_args()
mime = "image/png" if a.image.endswith(".png") else "image/jpeg"
url = f"data:{mime};base64," + base64.b64encode(open(a.image, "rb").read()).decode()
body = {
    "state": "A photo taken by a person walking around, from where they stand.",
    "images": [url],
    "questions": {
        "where": {"type": "choice", "instructions": "Looking at this photo, where should the person go next?",
                  "criteria": {k: k for k in ["Stand here", "Left", "Right", "Up", "Down"]}},
        "also": {"type": "choice", "instructions": "Looking at this photo, what should the person do here?",
                 "criteria": {k: k for k in ["Take a photo", "Sit down", "Go inside", "Keep walking"]}},
        "church": {"type": "noul", "instructions": "Is there a church or a tower in the photo?"},
        "beach": {"type": "noul", "instructions": "Is this photo taken on a beach?"},
    },
}
line = json.dumps(body) + "\n"
p = subprocess.run([a.cli, "-m", a.model, "--mmproj", a.mmproj, "-ngl", "0", "-t", a.threads, "--time"], input=line + line, capture_output=True, text=True)
if p.returncode != 0:
    print("cli failed:", p.returncode, p.stderr[-600:]); sys.exit(1)
outs = [json.loads(l) for l in p.stdout.strip().splitlines()]
mine = outs[0]
if "error" in mine:
    print("cli error:", mine["error"]); sys.exit(1)
print("cli:", mine["usage"], "wall ms first/second:", round(outs[0]["_wall_ms"]), round(outs[1]["_wall_ms"]))
theirs = None
if a.server:
    req = urllib.request.Request(a.server + "/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    theirs = json.loads(urllib.request.urlopen(req, timeout=300).read())
    if "error" in theirs:
        print("server error:", theirs["error"]); theirs = None
worst = 0.0
for qid, q in mine["answers"].items():
    def vec(ans):
        return [ans["noul"]] if ans["type"] == "noul" else [ans["probabilities"][k] for k in body["questions"][qid]["criteria"]]
    m = vec(q)
    line = f"  {qid:7} cli {[f'{x:.4f}' for x in m]}"
    if theirs:
        t = vec(theirs["answers"][qid]); d = max(abs(x - y) for x, y in zip(m, t)); worst = max(worst, d)
        line += f"  server {[f'{x:.4f}' for x in t]}  diff {d:.4f}"
    print(line)
    if q["type"] != "noul":
        print(f"          choice: {q['choice']}")
if theirs:
    print(f"largest difference: {worst:.4f}")
