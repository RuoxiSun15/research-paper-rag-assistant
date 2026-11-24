# Self-hosted Deployment

PaperLens follows a **Bring Your Own Azure** model. There is no shared hosted
backend and no shared Azure allowance. Each deployment uses the Azure resources
and billing account selected by its owner.

## What each user must provide

Create these resources in your own Azure subscription:

1. Azure AI Document Intelligence
2. Azure OpenAI or Microsoft Foundry chat deployment
3. Azure OpenAI embedding deployment (`text-embedding-3-small` recommended)
4. Azure AI Search with vector search support
5. Optional Application Insights and Azure AI Content Safety resources

Then copy `.env.example` to `.env` and enter only your own endpoints,
deployment names, and authentication settings.

```bash
cp .env.example .env
az login
az account set --subscription "<your-subscription-id>"
```

Never send Azure keys to the project maintainer and never enter credentials
into an untrusted hosted copy of PaperLens.

## Local deployment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. All resulting Azure usage is charged to the
subscription configured in your local `.env` or Azure CLI session.

## Deploying to your own Azure Web App

The repository serves the frontend from the same FastAPI application, so only
one web application is required. Create the App Service plan and Web App in
your own subscription, then configure the startup command as:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Store production credentials in Azure Key Vault or encrypted Web App settings.
Do not commit a production `.env` file. When using API keys in Web App settings,
set each `*_USE_AZURE_CLI` option to `false`. For a stronger production setup,
use Managed Identity and least-privilege Azure RBAC roles.

Deploy from your own fork or a local Git archive. Before making the URL public,
add authentication, authorization, rate limiting, restricted CORS, upload size
limits, and monitoring. The development API is not intended to be exposed
unchanged on the public internet.

## Cost controls

- Create an Azure Budget and 50%, 80%, and 100% alerts.
- Start with the smallest suitable service tiers and model quotas.
- Limit the number of figures sent to the vision model.
- Remove unused indexes, deployments, and hosting resources.
- Remember that Azure Budget alerts notify you but do not automatically stop
  resources.

## GitHub Pages

GitHub Pages can display a static project page, but it cannot run the PaperLens
Python backend. Publishing the current `frontend/` directory alone would leave
upload, retrieval, and question answering unavailable. A functional PaperLens
instance must be self-hosted with the user's own backend and Azure resources.
