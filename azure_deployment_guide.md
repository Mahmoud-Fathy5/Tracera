# 🚀 Tracera — Azure Deployment Guide

Complete step-by-step guide to deploy the Tracera deepfake detection web app on Microsoft Azure.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Prepare Your Project](#2-prepare-your-project)
3. [Method 1: Azure App Service (Recommended)](#3-method-1-azure-app-service-recommended)
4. [Method 2: Docker + Azure Container Instances](#4-method-2-docker--azure-container-instances)
5. [Method 3: Azure Virtual Machine](#5-method-3-azure-virtual-machine)
6. [Connect the Browser Extension API](#6-connect-the-browser-extension-api)
7. [Custom Domain & SSL](#7-custom-domain--ssl)
8. [Monitoring & Scaling](#8-monitoring--scaling)
9. [Cost Estimation](#9-cost-estimation)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Before starting, make sure you have:

| Requirement | How to Get It |
|---|---|
| **Azure Account** | Sign up at [portal.azure.com](https://portal.azure.com) — free tier available with $200 credit |
| **Azure CLI** | Install: `winget install Microsoft.AzureCLI` (Windows) or [download](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) |
| **Git** | Install: `winget install Git.Git` |
| **Python 3.10+** | Already installed (for local testing) |
| **Docker Desktop** (Method 2 only) | [Download](https://www.docker.com/products/docker-desktop/) |

### Login to Azure CLI

```powershell
# Login to your Azure account
az login

# Set your subscription (if you have multiple)
az account list --output table
az account set --subscription "Your-Subscription-Name"
```

---

## 2. Prepare Your Project

### 2.1 Fix the `requirements.txt`

> [!WARNING]
> Your current `requirements.txt` has a leading space on line 1. Remove it:

Open `requirements.txt` and make sure it looks exactly like this:

```text
flask>=3.0,<4.0
flask-cors>=4.0,<5.0
flask-limiter>=3.5,<4.0
Pillow>=10.0,<11.0
xgboost>=2.0,<3.0
scikit-learn>=1.3,<2.0
numpy>=1.24,<2.0
gunicorn>=22.0,<23.0

# PyTorch CPU-only (smaller download, no CUDA needed for inference)
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.0,<3.0
torchvision>=0.15,<1.0
```

### 2.2 Update CORS for Browser Extension

In `app.py`, update the CORS configuration to allow requests from the browser extension:

```python
# CORS — allow API access from browser extension and web
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

> [!NOTE]
> The `"origins": "*"` is already set in your code. This allows the browser extension to call your API from any webpage. For production, you could restrict to specific origins, but for extensions this is generally fine since the extension runs locally.

### 2.3 Create `startup.sh` (for Azure App Service)

Create a file called `startup.sh` in the project root:

```bash
#!/bin/bash
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 1 "app:create_app()"
```

> [!IMPORTANT]
> We use **1 worker** and a **600-second timeout** because the PyTorch model is heavy and needs time to load. Do NOT increase workers unless you have a high-memory tier (B3+ / P2+).

### 2.4 Create `.deployment` file

Create a file called `.deployment` in the project root:

```ini
[config]
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

### 2.5 Create `.gitignore`

```gitignore
__pycache__/
*.pyc
.env
*.ipynb
fat7y.png
smallMoMedhat.jpg
colors.jpg
```

---

## 3. Method 1: Azure App Service (Recommended)

> [!TIP]
> This is the easiest method. Azure handles infrastructure, scaling, SSL, and monitoring for you.

### Step 1: Create a Resource Group

```powershell
az group create --name tracera-rg --location eastus
```

### Step 2: Create an App Service Plan

```powershell
# B2 tier (recommended for ML models: 3.5 GB RAM, 2 vCPUs)
az appservice plan create `
    --name tracera-plan `
    --resource-group tracera-rg `
    --sku B2 `
    --is-linux
```

> [!IMPORTANT]
> **Minimum tier: B2 (3.5 GB RAM)**. The PyTorch VGG-16 model + XGBoost models require ~2-3 GB RAM at runtime. The Free (F1) and B1 tiers will crash with out-of-memory errors.

### Step 3: Create the Web App

```powershell
az webapp create `
    --name tracera-app `
    --resource-group tracera-rg `
    --plan tracera-plan `
    --runtime "PYTHON:3.11"
```

> [!NOTE]
> The name `tracera-app` must be globally unique. If it's taken, try `tracera-app-2026` or `tracera-yourname`.

### Step 4: Configure App Settings

```powershell
# Set startup command
az webapp config set `
    --name tracera-app `
    --resource-group tracera-rg `
    --startup-file "startup.sh"

# Set environment variables
az webapp config appsettings set `
    --name tracera-app `
    --resource-group tracera-rg `
    --settings `
        SCM_DO_BUILD_DURING_DEPLOYMENT=true `
        WEBSITES_PORT=8000 `
        PYTHON_ENABLE_GUNICORN_MULTIWORKERS=false
```

### Step 5: Configure Build for PyTorch

Azure's build system needs to know about the custom PyTorch index:

```powershell
# Increase build timeout (PyTorch is a large package)
az webapp config appsettings set `
    --name tracera-app `
    --resource-group tracera-rg `
    --settings `
        SCM_BUILD_TIMEOUT=1800 `
        PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
```

### Step 6: Deploy Your Code

**Option A: Deploy via ZIP (Simplest)**

```powershell
# Navigate to your project directory
cd "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2"

# Create a zip of everything needed (exclude unnecessary files)
Compress-Archive -Path app.py, inference.py, requirements.txt, startup.sh, .deployment, model, static -DestinationPath tracera-deploy.zip -Force

# Deploy the zip
az webapp deploy `
    --name tracera-app `
    --resource-group tracera-rg `
    --src-path tracera-deploy.zip `
    --type zip
```

**Option B: Deploy via Git (Better for ongoing updates)**

```powershell
# Initialize git repo
cd "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2"
git init
git add app.py inference.py requirements.txt startup.sh .deployment model/ static/
git commit -m "Initial Tracera deployment"

# Configure local git deployment
az webapp deployment source config-local-git `
    --name tracera-app `
    --resource-group tracera-rg

# The command above will output a Git URL like:
# https://tracera-app.scm.azurewebsites.net/tracera-app.git

# Add Azure as a remote and push
az webapp deployment list-publishing-credentials `
    --name tracera-app `
    --resource-group tracera-rg `
    --query "{username: publishingUserName, password: publishingPassword}" `
    --output table

# Add remote (replace URL from the config-local-git output)
git remote add azure <YOUR_GIT_URL>
git push azure master
```

### Step 7: Wait for Deployment & Check Logs

```powershell
# Stream live logs to see deployment progress
az webapp log tail `
    --name tracera-app `
    --resource-group tracera-rg

# Check deployment status
az webapp show `
    --name tracera-app `
    --resource-group tracera-rg `
    --query "state" `
    --output tsv
```

### Step 8: Test Your Deployment

```powershell
# Open the app in your browser
az webapp browse --name tracera-app --resource-group tracera-rg

# Or directly visit:
# https://tracera-app.azurewebsites.net

# Test the API health endpoint
curl https://tracera-app.azurewebsites.net/api/health
```

Expected response:
```json
{"model": "GramNet v3", "status": "ok"}
```

---

## 4. Method 2: Docker + Azure Container Instances

> [!TIP]
> Best if you want full control over the runtime environment and reproducible builds.

### Step 1: Create `Dockerfile`

Create this file in your project root:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py inference.py startup.sh ./
COPY model/ ./model/
COPY static/ ./static/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run with gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout=600", "--workers=1", "app:create_app()"]
```

### Step 2: Create Azure Container Registry (ACR)

```powershell
# Create resource group (if not already created)
az group create --name tracera-rg --location eastus

# Create container registry
az acr create `
    --name traceraregistry `
    --resource-group tracera-rg `
    --sku Basic `
    --admin-enabled true
```

### Step 3: Build & Push Docker Image

```powershell
# Login to ACR
az acr login --name traceraregistry

# Build image directly on Azure (no local Docker needed!)
az acr build `
    --registry traceraregistry `
    --image tracera:v1 `
    --file Dockerfile `
    "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2"
```

> [!NOTE]
> `az acr build` builds the image in the cloud, so you don't even need Docker Desktop installed locally. This is especially useful since the image will be large (~3-4 GB with PyTorch).

### Step 4: Deploy to Azure Container Instances

```powershell
# Get ACR credentials
$ACR_PASSWORD = az acr credential show `
    --name traceraregistry `
    --query "passwords[0].value" `
    --output tsv

# Create container instance
az container create `
    --name tracera-container `
    --resource-group tracera-rg `
    --image traceraregistry.azurecr.io/tracera:v1 `
    --registry-login-server traceraregistry.azurecr.io `
    --registry-username traceraregistry `
    --registry-password $ACR_PASSWORD `
    --cpu 2 `
    --memory 4 `
    --ports 8000 `
    --dns-name-label tracera-dns `
    --ip-address Public
```

### Step 5: Get Your Public URL

```powershell
az container show `
    --name tracera-container `
    --resource-group tracera-rg `
    --query "ipAddress.fqdn" `
    --output tsv
```

Your app will be at: `http://tracera-dns.eastus.azurecontainer.io:8000`

---

## 5. Method 3: Azure Virtual Machine

> [!NOTE]
> Best for full control and SSH access. More setup work but most flexible.

### Step 1: Create the VM

```powershell
az vm create `
    --name tracera-vm `
    --resource-group tracera-rg `
    --image Ubuntu2404 `
    --size Standard_B2ms `
    --admin-username azureuser `
    --generate-ssh-keys `
    --public-ip-sku Standard
```

### Step 2: Open Port 5000

```powershell
az vm open-port `
    --name tracera-vm `
    --resource-group tracera-rg `
    --port 5000 `
    --priority 1001

# Also open port 80 for nginx reverse proxy
az vm open-port `
    --name tracera-vm `
    --resource-group tracera-rg `
    --port 80 `
    --priority 1002
```

### Step 3: SSH Into the VM

```powershell
# Get public IP
$VM_IP = az vm show `
    --name tracera-vm `
    --resource-group tracera-rg `
    --show-details `
    --query publicIps `
    --output tsv

# SSH in
ssh azureuser@$VM_IP
```

### Step 4: Setup the VM (Run on the VM via SSH)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and pip
sudo apt install -y python3.11 python3.11-venv python3-pip nginx git

# Clone/upload your project (or use scp from your local machine)
mkdir ~/tracera && cd ~/tracera

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Upload Your Project Files (from your local machine)

```powershell
# Upload project files to VM
scp -r "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2\app.py" azureuser@${VM_IP}:~/tracera/
scp -r "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2\inference.py" azureuser@${VM_IP}:~/tracera/
scp -r "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2\requirements.txt" azureuser@${VM_IP}:~/tracera/
scp -r "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2\model" azureuser@${VM_IP}:~/tracera/
scp -r "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2\static" azureuser@${VM_IP}:~/tracera/
```

### Step 6: Configure Systemd Service (on VM)

```bash
sudo nano /etc/systemd/system/tracera.service
```

Paste this content:

```ini
[Unit]
Description=Tracera Deepfake Detection API
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/tracera
Environment="PATH=/home/azureuser/tracera/venv/bin"
ExecStart=/home/azureuser/tracera/venv/bin/gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 "app:create_app()"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable tracera
sudo systemctl start tracera

# Check status
sudo systemctl status tracera
```

### Step 7: Configure Nginx Reverse Proxy (on VM)

```bash
sudo nano /etc/nginx/sites-available/tracera
```

```nginx
server {
    listen 80;
    server_name _;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/tracera /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

Your app is now at: `http://<VM-PUBLIC-IP>/`

---

## 6. Connect the Browser Extension API

Once deployed, your API base URL will be one of:

| Method | URL |
|---|---|
| **App Service** | `https://tracera-app.azurewebsites.net` |
| **Container Instances** | `http://tracera-dns.eastus.azurecontainer.io:8000` |
| **Virtual Machine** | `http://<VM-PUBLIC-IP>` |

In the browser extension (created separately), set the `API_BASE_URL` in the extension's settings/config to your deployed URL.

> [!IMPORTANT]
> The browser extension sends images to `{API_BASE_URL}/api/predict` via POST with a `multipart/form-data` body containing the image file in the `image` field. The response JSON looks like:
> ```json
> {
>   "verdict": "Fake",
>   "confidence": 0.9234,
>   "attribution": "GAN",
>   "attribution_confidence": 0.8712
> }
> ```

---

## 7. Custom Domain & SSL

### For App Service:

```powershell
# Add custom domain
az webapp config hostname add `
    --webapp-name tracera-app `
    --resource-group tracera-rg `
    --hostname www.tracera.com

# Enable free managed SSL certificate
az webapp config ssl create `
    --name tracera-app `
    --resource-group tracera-rg `
    --hostname www.tracera.com

# Bind the certificate
az webapp config ssl bind `
    --name tracera-app `
    --resource-group tracera-rg `
    --certificate-thumbprint <THUMBPRINT> `
    --ssl-type SNI
```

### For VM (using Let's Encrypt):

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 8. Monitoring & Scaling

### Enable Application Insights

```powershell
# Create Application Insights
az monitor app-insights component create `
    --app tracera-insights `
    --resource-group tracera-rg `
    --location eastus

# Link to your webapp
az webapp config appsettings set `
    --name tracera-app `
    --resource-group tracera-rg `
    --settings `
        APPLICATIONINSIGHTS_CONNECTION_STRING="<YOUR_CONNECTION_STRING>"
```

### Auto-Scaling (App Service)

```powershell
# Scale up to more powerful tier
az appservice plan update `
    --name tracera-plan `
    --resource-group tracera-rg `
    --sku P1V3

# Enable auto-scale (scale out to max 3 instances based on CPU)
az monitor autoscale create `
    --resource-group tracera-rg `
    --resource tracera-plan `
    --resource-type Microsoft.Web/serverfarms `
    --min-count 1 `
    --max-count 3 `
    --count 1

az monitor autoscale rule create `
    --resource-group tracera-rg `
    --autoscale-name <autoscale-name> `
    --scale out 1 `
    --condition "CpuPercentage > 70 avg 5m"
```

---

## 9. Cost Estimation

| Tier | Monthly Cost (Est.) | RAM | vCPUs | Best For |
|---|---|---|---|---|
| **B2 App Service** | ~$55/mo | 3.5 GB | 2 | Development / Low traffic |
| **B3 App Service** | ~$110/mo | 7 GB | 4 | Medium traffic |
| **P1V3 App Service** | ~$140/mo | 8 GB | 2 | Production with auto-scale |
| **Container Instance** | ~$45-90/mo | 4 GB | 2 | Pay-per-second, no auto-scale |
| **B2ms VM** | ~$60/mo | 8 GB | 2 | Full control, SSH access |

> [!TIP]
> Start with **B2 App Service** for testing, then upgrade to **P1V3** for production. Azure free tier gives you **$200 credit** for the first 30 days.

---

## 10. Troubleshooting

### Common Issues

| Problem | Solution |
|---|---|
| **App crashes on startup** | Check logs: `az webapp log tail`. Usually an OOM issue — upgrade to B2+ tier |
| **502 Bad Gateway** | Gunicorn hasn't started yet. Wait 2-3 minutes for PyTorch model to load. Increase `--timeout` |
| **ModuleNotFoundError: torch** | PyTorch didn't install properly. Check `PIP_EXTRA_INDEX_URL` is set correctly |
| **Deployment timeout** | PyTorch is huge (~800 MB). Set `SCM_BUILD_TIMEOUT=1800` and retry |
| **File too large to deploy** | Model files are ~45 MB total. Use ZIP deploy, not Git for large files |
| **CORS error from extension** | Make sure `flask-cors` is configured with `origins: "*"` |
| **Rate limited (429)** | Your app has a 10 req/min limit. Adjust in `app.py` if needed |

### View Logs

```powershell
# Live logs
az webapp log tail --name tracera-app --resource-group tracera-rg

# Download log files
az webapp log download --name tracera-app --resource-group tracera-rg --log-file logs.zip
```

### SSH into App Service Container

```powershell
az webapp ssh --name tracera-app --resource-group tracera-rg
```

### Restart the App

```powershell
az webapp restart --name tracera-app --resource-group tracera-rg
```

---

## Quick Start Summary

For the fastest deployment, run these commands in order:

```powershell
# 1. Login
az login

# 2. Create resources
az group create --name tracera-rg --location eastus
az appservice plan create --name tracera-plan --resource-group tracera-rg --sku B2 --is-linux
az webapp create --name tracera-app --resource-group tracera-rg --plan tracera-plan --runtime "PYTHON:3.11"

# 3. Configure
az webapp config set --name tracera-app --resource-group tracera-rg --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 app:create_app()"
az webapp config appsettings set --name tracera-app --resource-group tracera-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000 SCM_BUILD_TIMEOUT=1800 PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# 4. Deploy
cd "C:\Users\MOHAMED MEDHAT\Downloads\web v2\web v2"
Compress-Archive -Path app.py,inference.py,requirements.txt,model,static -DestinationPath tracera-deploy.zip -Force
az webapp deploy --name tracera-app --resource-group tracera-rg --src-path tracera-deploy.zip --type zip

# 5. Test
az webapp browse --name tracera-app --resource-group tracera-rg
```

**Your Tracera app is now live at: `https://tracera-app.azurewebsites.net` 🎉**
