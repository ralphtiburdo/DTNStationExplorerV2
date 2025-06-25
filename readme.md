# DTN Station Explorer

## Setup (Local)

1. `git clone <repo-url>`  
2. `cd dtn-station-explorer`  
3. `python3 -m venv venv && source venv/bin/activate`  
4. `pip install -r requirements.txt`  
5. Create a `.env` file (see example in repo root).  
6. `streamlit run app.py`  

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.  
2. Go to https://share.streamlit.io → **New app** → connect your GitHub repo.  
3. Choose branch (`main`) and file (`app.py`).  
4. Under **Advanced settings → Secrets**, add:
   - `CLIENT_ID_INTERNAL`, `CLIENT_SECRET_INTERNAL`  
   - `CLIENT_ID_CORE`, `CLIENT_SECRET_CORE`  
   - `CLIENT_ID_PLUS`, `CLIENT_SECRET_PLUS`  
5. Click **Deploy**.  

Subsequent pushes to `main` will auto-redeploy.
