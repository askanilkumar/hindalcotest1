# Self-Test Simulation Lab — Mini Day 2 → 3 → 4

**Purpose:** a compressed, single-thread run of the whole production pipeline — train → export → spec → build → containerize → deploy → monitor → teardown — so you personally hit every friction point before 25 people do it live. Uses ONE worked example (the G.2 boiler-efficiency DNN) end to end. The image-classifier track (G.3) follows the identical Day-3/4 steps once you have a `.pt` file — no need to duplicate this run for it.

**Total time:** ~2–2.5 hours, most of it Azure provisioning wait time, not your active work.

**Tonight you are testing SOLO, not simulating 5–6 teams.** Everywhere the earlier draft said "your team's branch," this version uses one repo and the `main` branch, because it's just you. Section 0.0 below has a short note on what changes when you scale this up for the real room tomorrow.

**Correction to earlier notes:** the notebook's Section C.5 checkpoint cell already calls `torch.save(...)`. The real gap is that **G.2 and G.3 — the graded tasks — are blank `TODO` skeletons with no save step written in.** This lab shows you exactly what to add.

---

## How to read this document

Every step is tagged with WHERE you do it:

- 📓 **COLAB** — in the `D2_Module4_DL_Foundations_Lab` notebook, in your browser
- 💻 **VM TERMINAL** — a terminal window on the Linux VM
- 🧩 **CLINE** — the Cline chat panel inside VS Code
- 🌐 **GITHUB.COM** — GitHub's website, in your browser
- 🖱️ **AZURE PORTAL** — [ml.azure.com](https://ml.azure.com), in your browser

And every code block is tagged with WHAT TO DO with it:

- **[RUN AS-IS]** — already exists in the notebook exactly like this; just click the ▶ play button on that cell. Don't type or paste anything.
- **[REPLACE CELL CONTENTS]** — click inside the existing cell, select everything in it (Ctrl+A while your cursor is in the cell), delete, then paste the block given.
- **[NEW CELL]** — this code does not exist yet; add a new cell in the location described, then paste the block into it.
- **[TYPE IN TERMINAL]** — type or paste into a terminal window and press Enter.
- **[PASTE INTO CLINE CHAT]** — paste as a chat message to Cline, not into a code file.

---

## Part 0.0 — One-time GitHub setup (10 min, do this first, before Part 0's checklist)

📓/🌐 You need one GitHub repository and one access token before Colab can push anything.

1. 🌐 **GITHUB.COM** — log in (or create a free account if you don't have one).
2. 🌐 Click the **+** icon, top right of any GitHub page → **New repository**.
3. 🌐 Name it `hindalco-boiler-app-test`. Set it to **Private**. Leave everything else default. Click **Create repository**. You now have an empty repo — you do NOT need to manually create `models/`, `specs/`, or `tasks/` folders; the commands later create them automatically.
4. 🌐 Generate a token: click your profile picture (top right) → **Settings** → scroll to the very bottom of the left-hand menu → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token** → **Generate new token (classic)**.
5. 🌐 Give it a name like `hindalco-test`, set an expiration (7 days is fine), tick the **`repo`** checkbox (this grants read/write to your repos), scroll down → **Generate token**.
6. 🌐 **Copy the token now** — GitHub shows it exactly once. Paste it somewhere safe temporarily (a text file) — you'll need it in step 2 of Part 1.

**Scaling to the real room tomorrow (read, don't do yet):** for 25 participants, create one shared repo, one branch per team (e.g. `team-mining-mavericks`), and one token per team scoped to that repo only — hand out each team's token on a slip of paper or a private message, not projected on screen. Everything below still applies per-team, just swap `main` for the team's branch name.

---

## Part 0 — Prereqs checklist (5 min)

- [ ] `hindalco-boiler-app-test` GitHub repo created, token copied (Part 0.0 above)
- [ ] 📓 Google account open, `D2_Module4_DL_Foundations_Lab` notebook open in Colab
- [ ] 💻 Linux VM reachable, VS Code installed, Cline extension installed in VS Code (Extensions icon in VS Code's left sidebar → search "Cline" → Install, if not already there)
- [ ] OpenAI API key in hand, budgeted for `gpt-4o-mini`
- [ ] 🖱️ Azure login for the $10-credit account, confirm you can reach [ml.azure.com](https://ml.azure.com) and see (or can create) a workspace
- [ ] 💻 **VM TERMINAL** — confirm tools are installed: type each of these, press Enter, confirm each prints a version instead of "command not found":
  ```
  git --version
  docker --version
  python3 --version
  az --version
  ```
  If `az` (Azure CLI) is missing: `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash` (Debian/Ubuntu-based VM — if your VM is a different Linux flavor, search "install azure cli" + your distro name).

---

## Part 1 — Mini Day 2: Train + Export (25–30 min, all steps in Colab unless noted)

### Step 0.5 — Open the notebook in Colab (2 min, do this before step 1 below)

- Your copy of the notebook in the Hindalco folder is named `D2_Module4_DL_Foundations_Lab` with **no `.ipynb` extension**. Rename your local copy to `D2_Module4_DL_Foundations_Lab.ipynb` first — Colab's upload picker may not recognize it otherwise.
- 📓 Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → choose that renamed file. This opens all 82 cells as a live notebook, nothing pre-run.
- **Tip:** after running a cell with Shift+Enter (instead of clicking ▶), Colab runs that cell AND auto-selects the next one — so you can keep pressing Shift+Enter to move through cells in order without re-clicking each one. Stop doing this once you reach the C.5 cell (step 3 below) and switch to the outline-view jump instead, so you don't accidentally run Sections D/E/F.

**Do not use Colab's "Runtime → Run all."** That runs all 82 cells including Sections D/E/F (images, transformer theory) which you don't need tonight and which take real GPU time. Instead:

1. 📓 In the notebook, find the code cell whose first comment line reads `# ── Step 0: imports, seeds and device`. Click on it, then click the **▶** play button, or press Shift+Enter, to run it. **[RUN AS-IS]**
2. 📓 Use Colab's outline view to jump around instead of scrolling: click the ☰ (lines) icon in the top-left of Colab's left sidebar — this shows every markdown heading as a clickable list. Click through the headings in order (`A.1`, `A.2`, `A.3`, … up to `C.5`) and run each code cell under them with ▶, top to bottom, in order. **[RUN AS-IS, one cell at a time, in order]** This trains a few small demo models along the way (Sections A/B/C) — that's expected and harmless, they're the shared building blocks G.2 reuses. This takes a few minutes total, not long.
3. 📓 Stop once you've run the cell whose comment reads `# ── C.5 Final model, business threshold, honest test evaluation` (it ends by printing `Saved → anode_effect_dnn.pt` — that's a different model, not today's target, ignore it).
4. 📓 In the outline view, click ahead to the heading **`### G.2 Hands-on task 1 — Boiler efficiency regression DNN (40%)`**. Skip everything between C.5 and here (Sections D, E, F) — do not run those cells tonight.
5. 📓 Run the code cell right under that heading — its comment reads `# ── G.2 Boiler efficiency data + baseline (run as-is)`. **[RUN AS-IS]** This creates the variables `boiler_df`, `Xb`, `yb`, `Xb_tr`, `Xb_va`, `Xb_te`, `b_scaler` that the next step needs.
6. 📓 The next cell down is the one with comment `# ── G.2 YOUR SOLUTION`, currently containing only `boiler_model = None   # TODO` and a print statement. Click inside THIS cell, press Ctrl+A (selects everything in that cell only), press Delete, then paste the block below in its place. **[REPLACE CELL CONTENTS]** Run it with ▶ once pasted — training takes under a minute.

```python
# Scale features and target
Xs_tr, Xs_va, Xs_te = b_scaler.transform(Xb_tr), b_scaler.transform(Xb_va), b_scaler.transform(Xb_te)
y_mu, y_sd = yb_tr.mean(), yb_tr.std()
ys_tr, ys_va = (yb_tr - y_mu) / y_sd, (yb_va - y_mu) / y_sd

Xt_tr_b = torch.tensor(Xs_tr, dtype=torch.float32)
yt_tr_b = torch.tensor(ys_tr, dtype=torch.float32)
Xt_va_b = torch.tensor(Xs_va, dtype=torch.float32).to(DEVICE)
yt_va_b = torch.tensor(ys_va, dtype=torch.float32).to(DEVICE)
boiler_loader = make_loader(Xt_tr_b, yt_tr_b, batch_size=128)

boiler_model = nn.Sequential(
    nn.Linear(Xs_tr.shape[1], 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
    nn.Linear(64, 32), nn.GELU(),
    nn.Linear(32, 1)
).to(DEVICE)

opt = torch.optim.AdamW(boiler_model.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = nn.MSELoss()

best_val, best_state, bad, patience = float("inf"), None, 0, 10
for epoch in range(150):
    boiler_model.train()
    for xb, yb_ in boiler_loader:
        xb, yb_ = xb.to(DEVICE), yb_.to(DEVICE)
        opt.zero_grad()
        loss_fn(boiler_model(xb).squeeze(1), yb_).backward()
        opt.step()
    boiler_model.eval()
    with torch.no_grad():
        val_loss = loss_fn(boiler_model(Xt_va_b).squeeze(1), yt_va_b).item()
    if val_loss < best_val:
        best_val, best_state, bad = val_loss, copy.deepcopy(boiler_model.state_dict()), 0
    else:
        bad += 1
        if bad >= patience:
            break
boiler_model.load_state_dict(best_state)

with torch.no_grad():
    pred_te = (boiler_model(torch.tensor(Xs_te, dtype=torch.float32).to(DEVICE)).squeeze(1).cpu().numpy() * y_sd) + y_mu
test_mae = mean_absolute_error(yb_te, pred_te)
print(f"Test MAE: {test_mae:.3f} percentage points (target ≤ 0.35)")
```

You should see something like `Test MAE: 0.21...` printed — well under the 0.35 target. If you see an error instead, the most likely cause is running step 6 before step 5 (the baseline cell) — go back and run step 5 first.

7. 📓 Now add a **brand-new cell right after** the one you just ran: hover your mouse just below that cell until a thin **+ Code** button appears, click it, and paste this block into the new empty cell. **[NEW CELL]** Run it with ▶.

```python
import joblib
torch.save(boiler_model.state_dict(), "boiler_efficiency_dnn.pt")
joblib.dump(b_scaler, "boiler_efficiency_scaler.pkl")
joblib.dump({"y_mu": float(y_mu), "y_sd": float(y_sd), "features": list(Xb.columns)},
            "boiler_efficiency_meta.pkl")
print("Exported: boiler_efficiency_dnn.pt, boiler_efficiency_scaler.pkl, boiler_efficiency_meta.pkl")
```

8. 📓 Before the next cell will work, store your GitHub token as a Colab secret (one-time, per Colab session/browser — it does NOT get saved into the notebook file itself): click the **🔑 key icon** in Colab's LEFT sidebar (below the folder icon) → **+ Add new secret** → Name: `GITHUB_TOKEN` → Value: paste the token you copied in Part 0.0 → make sure the **toggle switch is ON** so this notebook can access it.
9. 📓 Add **another new cell** right after (same **+ Code** method as step 7), paste this, and run it. **[NEW CELL]** Replace `<your-username>` with your actual GitHub username.

```python
from google.colab import userdata
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
REPO = "github.com/<your-username>/hindalco-boiler-app-test.git"
BRANCH = "main"

!git clone https://{GITHUB_TOKEN}@{REPO} repo
!mkdir -p repo/models
!cp boiler_efficiency_dnn.pt boiler_efficiency_scaler.pkl boiler_efficiency_meta.pkl repo/models/
%cd repo
!git config user.email "you@hindalco-training.local"
!git config user.name "Anil"
!git add models/ && git commit -m "Add exported boiler efficiency model" && git push origin {BRANCH}
%cd ..
```

10. 🌐 **GITHUB.COM** — open `github.com/<your-username>/hindalco-boiler-app-test` in a browser tab, click into the `models` folder, confirm you see the 3 files. This is the actual proof the push worked, not just "no error printed."

**✅ Checkpoint:** the model artifact is in the GitHub repo, visible on the website, not stuck in a Downloads folder.

---

## Part 2 — Mini Day 3: Spec-Driven Build with Cline (40–50 min)

### 2.1 Clone the repo onto the VM (5 min)

11. 💻 **VM TERMINAL** — open a terminal on the Linux VM (not Colab, not your local laptop — the VM, since that's where VS Code + Cline are). **[TYPE IN TERMINAL]**:
```
git clone https://github.com/<your-username>/hindalco-boiler-app-test.git
cd hindalco-boiler-app-test
ls models/
```
The `ls` should list the 3 files from Colab. If it's empty, the push in step 9 didn't work — go back and check.

12. 💻 **[TYPE IN TERMINAL]**, still in the same folder:
```
mkdir -p specs tasks
```

### 2.2 Create the 4 spec files (10 min)

13. 💻 Using VS Code (open the folder: `File → Open Folder → hindalco-boiler-app-test`) or any text editor on the VM, create these 4 NEW files with these EXACT names and contents. **[NEW FILE]** for each.

**`specs/product.md`**
```markdown
# Product

Predict boiler efficiency (%) from live operating conditions so plant operators
can run what-if checks (e.g. "what if we raise excess O2 by 0.5%?") without
waiting for an engineer. Users: shift operators, process engineers.
```

**`specs/features.md`**
```markdown
# Features

## Predict Efficiency
- User enters or uploads operating conditions (load %, excess O2 %, coal GCV,
  coal moisture %, coal ash %, hours since soot blowing, ambient temp, flue gas temp, unit).
- App returns predicted boiler efficiency % and shows it against the
  linear-baseline number for comparison.
```

**`specs/api.md`**
```markdown
# API — POST /predict

Request JSON:
{
  "load_pct": float,
  "excess_o2_pct": float,
  "coal_gcv_kcal_kg": float,
  "coal_moisture_pct": float,
  "coal_ash_pct": float,
  "hrs_since_soot_blow": float,
  "ambient_temp_C": float,
  "flue_gas_temp_C": float,
  "unit": "CPP-1" | "CPP-2" | "CPP-3"
}

Response 200:
{ "predicted_efficiency_pct": float }

Errors:
- 400 — missing required field, names the field
- 422 — value out of physically sensible range (e.g. excess_o2_pct outside 0.5–8.0)
- 500 — model file failed to load
```
*(Note: `flue_gas_temp_C` IS a required model feature — verified by actually running this pipeline. Easy to miss since it reads like a derived value rather than something an operator types in; treat it as a real sensor reading here.)*

**`specs/architecture.md`**
```markdown
# Architecture

Streamlit UI → FastAPI service (main.py) → loads model.pt + scaler.pkl + meta.pkl
from models/ at startup → returns prediction. Packaged with a Dockerfile.
No database, no auth — out of scope for this test.
```

### 2.3 Create the 4 task files (5 min)

14. 💻 Create these 4 NEW files, one instruction each — this is the actual text you'll hand to Cline one at a time in section 2.4, not documentation. **[NEW FILE]** for each.

**`tasks/task-001.md`**
```markdown
Create a FastAPI app in main.py with a single GET /health endpoint that
returns {"status": "ok"}. No model logic yet. Include a requirements.txt
with fastapi and uvicorn.
```

**`tasks/task-002.md`**
```markdown
Add POST /predict to main.py per specs/api.md. Load boiler_efficiency_dnn.pt
(a PyTorch nn.Sequential: Linear(n,64)-BatchNorm1d-GELU-Dropout(0.1)-
Linear(64,32)-GELU-Linear(32,1)), boiler_efficiency_scaler.pkl, and
boiler_efficiency_meta.pkl from the models/ folder at startup. One-hot
encode the "unit" field to match meta.pkl's saved "features" list before
scaling. Un-scale the model's output using y_mu and y_sd from meta.pkl
before returning it. Add torch, joblib, scikit-learn to requirements.txt.
```

**`tasks/task-003.md`**
```markdown
Add input validation to the /predict endpoint per specs/api.md's Errors
section — return exactly the status codes listed there (400 for a missing
field naming which one, 422 for an out-of-range value, 500 if the model
failed to load at startup).
```

**`tasks/task-004.md`**
```markdown
Write a Dockerfile: python:3.11-slim base image, install requirements.txt,
copy the app code and the models/ folder into the image, run the app with
uvicorn on port 8000.
```

### 2.4 Build it with Cline (20 min)

15. 💻 **VM TERMINAL** — open VS Code on the folder if you haven't: type `code .` in the terminal while inside the `hindalco-boiler-app-test` folder (or open VS Code manually and use File → Open Folder).
16. 🧩 In VS Code, click the **Cline icon** in the left sidebar (installed in Part 0's checklist) to open its chat panel. Click the **gear/settings icon** inside Cline's panel → set **API Provider** to `OpenAI` → paste your OpenAI API key into the key field → set **Model** to `gpt-4o-mini` → save/close settings.
17. 🧩 Open `tasks/task-001.md` in the VS Code editor tab, select all its text (Ctrl+A) and copy (Ctrl+C). Click into Cline's chat input box at the bottom of its panel, paste (Ctrl+V), press Enter to send. **[PASTE INTO CLINE CHAT]** Cline will propose creating/editing files — click the **Approve** (or similarly labelled) button it shows for each file change.
18. 💻 Once Cline finishes, open a terminal INSIDE VS Code (menu: **Terminal → New Terminal**, not a separate window) and run: **[TYPE IN TERMINAL]**
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Leave this running. Open a SECOND terminal tab (the **+** icon next to the terminal tabs) to test it: **[TYPE IN TERMINAL]**
```
curl localhost:8000/health
```
You should see `{"status":"ok"}`. If not, tell Cline the error you see and let it fix it before moving on — don't move to task-002 with a broken task-001.

19. 🧩 Repeat the same copy-paste-send pattern for `tasks/task-002.md`. Once Cline finishes, in the terminal running uvicorn, press Ctrl+C to stop it, then restart it (`uvicorn main:app --reload`) so it picks up the new code, and test in your second terminal: **[TYPE IN TERMINAL]**
```
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "load_pct": 85, "excess_o2_pct": 3.2, "coal_gcv_kcal_kg": 3600,
  "coal_moisture_pct": 12, "coal_ash_pct": 38, "hrs_since_soot_blow": 10,
  "ambient_temp_C": 30, "flue_gas_temp_C": 150, "unit": "CPP-1"
}'
```
You should get back something like `{"predicted_efficiency_pct": 85.6}`.

20. 🧩 Repeat for `tasks/task-003.md`. Test the validation by removing one field from the curl payload above (e.g. delete the `"unit": "CPP-1"` part) and re-running — confirm you get a 400 error naming the missing field, not a raw Python traceback.
21. 🧩 Repeat for `tasks/task-004.md`. Once Cline finishes, in the VS Code terminal: **[TYPE IN TERMINAL]**
```
docker build -t boiler-app .
docker run -p 8000:8000 boiler-app
```
In your second terminal tab, run the exact same curl command from step 19 again — confirm you get the same kind of prediction back, now from the container instead of the plain `uvicorn` process.

22. 💻 **[TYPE IN TERMINAL]**, back in the first terminal (stop the running container first with Ctrl+C if needed):
```
git add .
git commit -m "Day 3: FastAPI app built with Cline"
git push origin main
```

**✅ Checkpoint:** container runs locally, returns a sane efficiency number, and the app code is pushed to GitHub. This is what Day 4 re-points at a live endpoint.

---

## Part 3 — Mini Day 4: MLflow → Azure ML → Deploy → Test → Teardown (30–50 min + provisioning wait)

**Do this through the Azure ML Studio UI (ml.azure.com) wherever possible, not CLI/YAML** — that's what participants will click tomorrow, and this is exactly where provisioning delays and unclear errors show up.

### 3.1 Log the model with MLflow (VM terminal — reuses the files already in your repo, no need to go back to Colab)

23. 🖱️ **AZURE PORTAL** — go to [ml.azure.com](https://ml.azure.com), open your workspace (or create one if none exists yet: **+ New workspace**, smallest/default settings). On the workspace's **Overview** page, find the field labeled **MLflow tracking URI** and click its copy icon.
24. 💻 **VM TERMINAL**, inside the `hindalco-boiler-app-test` folder: **[TYPE IN TERMINAL]**
```
pip install mlflow azureml-mlflow torch scikit-learn joblib
az login
```
`az login` prints a URL and a code — open the URL in a browser, enter the code, sign in with the same account as your Azure ML workspace. Come back to the terminal once it says you're logged in.

25. 💻 Create a new file `log_model.py` in the same folder with this content — it reloads the exact model you trained in Colab from the files in `models/`, so you don't need to retrain anything. **[NEW FILE]** Replace `<paste your MLflow tracking URI here>` with what you copied in step 23.

```python
import torch, torch.nn as nn, joblib, mlflow, mlflow.pytorch

meta = joblib.load("models/boiler_efficiency_meta.pkl")
model = nn.Sequential(
    nn.Linear(len(meta["features"]), 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
    nn.Linear(64, 32), nn.GELU(),
    nn.Linear(32, 1)
)
model.load_state_dict(torch.load("models/boiler_efficiency_dnn.pt", map_location="cpu"))
model.eval()

mlflow.set_tracking_uri("<paste your MLflow tracking URI here>")
mlflow.set_experiment("boiler-efficiency-test")
with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
    print("Logged to MLflow — check the Azure ML Studio Jobs tab now.")
```

26. 💻 **[TYPE IN TERMINAL]**:
```
python3 log_model.py
```

### 3.2 Register and deploy (Azure Portal clicks)

27. 🖱️ In Studio: left sidebar → **Jobs** → find the run `log_model.py` just created (it'll be under the `boiler-efficiency-test` experiment) → open it → **Models** tab → **Register model** → name it `boiler-efficiency-dnn` → confirm.
28. 🖱️ Left sidebar → **Endpoints** → **Real-time endpoints** → **+ Create** → give it a name → on the compute step, pick the **smallest CPU SKU available** (e.g. `Standard_DS2_v2` — avoid anything with "GPU" or a high number in the name) → attach the model you just registered → **Deploy**.
29. 🖱️ **Time this step with your phone or a stopwatch, starting from clicking Deploy.** Provisioning can take anywhere from ~3 to 15+ minutes — write down the actual number, it's what sets tomorrow's per-team time slot.
30. 🖱️ Wait for the deployment status to say **"Healthy."** If it instead says "Unhealthy" or "Failed," click into the deployment's **Logs** tab and read the error there — this is exactly the failure mode to practice reading calmly before a room is watching you do it live.
31. 🖱️ Once healthy, click the endpoint's **Test** tab, paste the same JSON payload used in the curl command from step 19, click **Test**, confirm a prediction comes back. Also copy the **REST endpoint URL** and **Key** shown on the **Consume** tab, and from your VM terminal: **[TYPE IN TERMINAL]**
```
curl -X POST "<paste REST endpoint URL>" -H "Authorization: Bearer <paste key>" -H "Content-Type: application/json" -d '{ ... same payload ... }'
```
32. 🖱️ Click the **Monitoring** tab briefly — note where metrics/logs show up, this is what you'll point to in Module 8/9 tomorrow.
33. 🖱️ **Delete the endpoint now** — go back to **Endpoints**, select it, click **Delete**, confirm. Don't leave it running after your test.
34. 🖱️ Left sidebar of the main Azure Portal (portal.azure.com, not ml.azure.com) → **Cost Management + Billing** → check actual spend for today — should be a small fraction of $1.

**✅ Checkpoint:** full loop closed — a model trained in Colab is now provably deployable and testable through the exact tools/flow tomorrow's Day 4 uses, and you have a real timing number for provisioning.

---

## Part 4 (optional, if time allows) — Azure DevOps CI/CD smoke test

35. 🌐 Create a free Azure DevOps organization + project at [dev.azure.com](https://dev.azure.com) if you don't have one.
36. 🖱️ Inside the project: **Project Settings → Service connections → New service connection → GitHub** → authorize it against `hindalco-boiler-app-test` — this connects to the SAME repo, it does not copy it into Azure Repos.
37. 💻 Create a new file at the repo root named `azure-pipelines.yml`: **[NEW FILE]**
```yaml
trigger: [main]
pool: { vmImage: 'ubuntu-latest' }
steps:
  - script: pip install -r requirements.txt pytest
  - script: pytest || echo "no tests yet — placeholder pass"
```
38. 💻 **[TYPE IN TERMINAL]**:
```
git add azure-pipelines.yml
git commit -m "Add CI pipeline"
git push origin main
```
39. 🖱️ In Azure DevOps: **Pipelines → New pipeline** → point it at the GitHub repo and the file above → confirm it fires automatically on the push and goes green. Full auto-redeploy-on-push is a stretch goal, not required for tomorrow — proving the trigger fires is enough tonight.

---

## You're ready when…

- [ ] Boiler model exported to `.pt` + scaler + meta, visible in your GitHub repo's `models/` folder on the website
- [ ] FastAPI `/predict` works locally against the exported artifact, tested with curl
- [ ] Docker container builds, runs, and responds correctly to the same curl command
- [ ] App code pushed to GitHub
- [ ] Azure ML endpoint deployed, tested with curl, deleted — and you've timed the deploy step with a real number
- [ ] Actual Azure spend checked in Cost Management and it's a small fraction of $10
- [ ] You have a real number for "how long does one team's deploy-test-delete window need to be" tomorrow

---
*Companion to `Hindalco_Day2-4_Revised_Approach_ProductionFirst.md` and the project's `program-delivery-design.md`.*
