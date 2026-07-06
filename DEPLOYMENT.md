# Deployment Guide

Two services to deploy:
1. **FastAPI backend** → Render (free tier)
2. **Streamlit front-end** → Streamlit Cloud (free tier)

---

## Pre-deployment checklist (do this locally first)

These steps must be completed on your machine before pushing to GitHub,
because the trained model and FAISS index are committed to the repo
(Render doesn't re-train on deploy).

```bash
# 1. Place listings.csv in data/
#    Download from: http://insideairbnb.com/get-the-data/ → New York City

# 2. Train XGBoost model + save encoders
python train_and_save_model.py
# → creates models/xgboost_model.joblib and models/encoders.joblib

# 3. Fetch Airbnb Help Center docs + build FAISS index
python ingest_policies.py
# → creates data/policies/*.txt and vector_store/faiss_index/

# 4. Verify locally
cp .env.example .env      # add your GROQ_API_KEY
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/health — should return {"status":"ok","model_loaded":true,...}
```

---

## Part 1 — Push to GitHub

```bash
cd airbnb-ai-service

git init
git add .
git commit -m "Initial commit — FastAPI + LangGraph Airbnb AI Service"

# Create repo at github.com/robertciceroson/airbnb-ai-service (New → Public)
git remote add origin https://github.com/robertciceroson/airbnb-ai-service.git
git branch -M main
git push -u origin main
```

> **Verify:** Go to your GitHub repo and confirm these folders are present:
> `models/`, `vector_store/`, `data/policies/`
> If they're missing, the model wasn't trained before pushing.

---

## Part 2 — Deploy FastAPI to Render

### Step 1: Create account
Go to [render.com](https://render.com) → Sign up with GitHub (recommended — enables auto-deploy on push).

### Step 2: New Web Service
- Dashboard → **New +** → **Web Service**
- Connect your GitHub account → select `airbnb-ai-service` repo
- Click **Connect**

### Step 3: Configure the service
Render will auto-detect `render.yaml`. If not, set manually:

| Field | Value |
|---|---|
| Name | `airbnb-ai-service` |
| Region | Ohio (US East) |
| Branch | `main` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Plan | **Free** |

### Step 4: Add environment variable
- Scroll to **Environment Variables**
- Add: `GROQ_API_KEY` = `gsk_your_actual_key_here`
- Click **Save Changes**

### Step 5: Deploy
- Click **Create Web Service**
- Wait 3–5 minutes for the build to complete
- Your API URL will be: `https://airbnb-ai-service.onrender.com`

### Step 6: Test the live API
```bash
# Health check
curl https://airbnb-ai-service.onrender.com/health

# Price prediction
curl -X POST https://airbnb-ai-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"borough":"Manhattan","neighbourhood":"Midtown","room_type":"Entire home/apt","checkin_month":7}'

# Chat
curl -X POST https://airbnb-ai-service.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the cancellation policy for a flexible listing?","conversation_id":"test-123","history":[]}'
```

> **Note:** Free tier Render services spin down after 15 minutes of inactivity.
> First request after inactivity takes ~30 seconds (cold start). This is normal for demo/portfolio use.

---

## Part 3 — Deploy Streamlit Front-end to Streamlit Cloud

### Step 1: Push streamlit_app.py to the same repo (already there)

### Step 2: Go to Streamlit Cloud
[share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub → **New app**

### Step 3: Configure
| Field | Value |
|---|---|
| Repository | `robertciceroson/airbnb-ai-service` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |

### Step 4: Add secret
- Click **Advanced settings** before deploying
- Under **Secrets**, add:
```toml
API_BASE_URL = "https://airbnb-ai-service.onrender.com"
```
- Click **Save** → **Deploy**

### Step 5: Live URLs
- Streamlit app: `https://airbnb-ai-service-[hash].streamlit.app`
- FastAPI docs: `https://airbnb-ai-service.onrender.com/docs`

---

## Updating after changes

```bash
# Make changes locally → test → push
git add .
git commit -m "Update: describe your change"
git push

# Render auto-deploys on push (takes ~2 min)
# Streamlit Cloud auto-deploys on push (takes ~1 min)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Render build fails | Check build logs — usually a missing package in requirements.txt |
| `/health` returns `model_loaded: false` | `models/` folder wasn't committed — re-run `train_and_save_model.py` locally and push |
| `/health` returns `vector_store_loaded: false` | `vector_store/` folder wasn't committed — re-run `ingest_policies.py` locally and push |
| Streamlit chat gets no response | Check `API_BASE_URL` secret is set correctly in Streamlit Cloud |
| Render cold start timeout | First request after inactivity is slow — normal on free tier |
