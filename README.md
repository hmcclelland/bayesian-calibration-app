# 🔬 Bayesian Calibration & Inverse Prediction Tool

Define any calibration equation, fit it to data with Bayesian MCMC (PyMC), and predict **X** from new **Y** values with full uncertainty quantification.

## 🌐 Use the Web App

The public web version is hosted at:

**�� [https://your-username.streamlit.app](https://your-username.streamlit.app)**

*(Update this link after deploying — see Deployment section below)*

The web version uses manual data entry only (no file uploads) for security.

---

## 🖥️ Run Locally (with CSV upload support)

Running locally enables CSV file uploads and runs entirely on your own machine.

### Option A: One-click setup (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/bayesian-calibration-app.git
cd bayesian-calibration-app
chmod +x setup.sh
./setup.sh
```

This will automatically:
1. Create a conda environment with all dependencies
2. Enable CSV upload mode
3. Open the app in your browser at http://localhost:8501

### Option B: Manual setup

**Prerequisites:** [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/bayesian-calibration-app.git
cd bayesian-calibration-app

# 2. Create conda environment
conda create -n bayes-cal python=3.11 -y
conda activate bayes-cal

# 3. Install Python packages
pip install -r requirements.txt

# 4. Enable local mode (allows CSV uploads)
# Open app_config.py and change:  MODE = "local"

# 5. Launch the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
bayesian-calibration-app/
├── app.py               # Main Streamlit app
├── app_config.py        # MODE = "cloud" or "local"
├── equation_engine.py   # SymPy symbolic equation parser + PyMC model builder
├── description.md       # ← Edit this to change the description on the page
├── requirements.txt     # Python dependencies
├── setup.sh             # One-click local setup script
└── .streamlit/
    └── config.toml      # Streamlit theme config
```

### Customising the page description

Edit **`description.md`** with any text editor. It supports full Markdown:

```markdown
Welcome to our **calibration tool**.

- Point one
- Point two

[Link text](https://example.com)
```

Changes appear on the next page refresh — no code edits needed.

---

## ☁️ Deploy to Streamlit Community Cloud

This gives you a free public URL like `https://your-app.streamlit.app`.

### Step 1 — Push to GitHub

Create a new repository on [github.com](https://github.com/new), then:

```bash
cd bayesian-calibration-app
git add -A
git commit -m "Bayesian calibration app"
git remote add origin https://github.com/YOUR_USERNAME/bayesian-calibration-app.git
git branch -M main
git push -u origin main
```

### Step 2 — Make sure `app_config.py` is set to cloud mode

```python
MODE = "cloud"
```

This disables file uploads on the public web version.

### Step 3 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch `main`, main file `app.py`
5. Click **Deploy**

The first deploy takes ~2-3 minutes while PyMC installs. Subsequent loads are fast.

### Step 4 — Update the link

Once deployed, update the URL at the top of this README.

---

## 🔒 Security Notes

- **Cloud mode** (`MODE = "cloud"`): File uploads are completely disabled. Users can only enter data via text fields. No files ever touch the server.
- **Local mode** (`MODE = "local"`): CSV uploads are enabled because the app runs on the user's own machine.
- **Streamlit Cloud** runs your app in an isolated container. It has read-only access to this single GitHub repo — not your other repos, not your personal data, not your organisation's systems.
- **Visitors** to your app URL see only the running web app. They cannot access your GitHub account or repo contents through the app.

---

## 📖 How It Works

1. **Type any equation** like `y = a + b*x` or `y = a * exp(b*x) + c`
2. **SymPy** parses it, identifies parameters, renders LaTeX, and builds a PyMC model automatically
3. **PyMC** (NUTS / Hamiltonian Monte Carlo) samples the full posterior distribution of all parameters
4. **Inverse prediction**: for each new Y value and each posterior draw, the equation is inverted (symbolically if possible, numerically otherwise) to get a distribution over X
5. **Credible intervals** are read from quantiles of the resulting X distribution
