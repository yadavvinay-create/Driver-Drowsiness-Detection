# Driver Drowsiness Detection System

A non-intrusive system that monitors driver eye state (awake / sleepy) in real time and issues alerts to prevent accidents. The project includes a Streamlit web UI with optional Progressive Web App (PWA) support and pre-trained models.

## Quick Links
- App (Streamlit): [streamlit_app/streamlit_app.py](streamlit_app/streamlit_app.py) — includes [`DrowsinessDetector`](streamlit_app/streamlit_app.py)  
- PWA-enabled app: [streamlit_app/streamlit_app_pwa.py](streamlit_app/streamlit_app_pwa.py)  
- PWA manifest & assets: [streamlit_app/manifest.json](streamlit_app/manifest.json), [streamlit_app/sw.js](streamlit_app/sw.js), [streamlit_app/offline.html](streamlit_app/offline.html), [streamlit_app/.streamlit/config.toml](streamlit_app/.streamlit/config.toml)  
- Icon generator: [streamlit_app/generate_icons.py](streamlit_app/generate_icons.py) — see [`create_icon`](streamlit_app/generate_icons.py) and [`generate_all_icons`](streamlit_app/generate_icons.py)  
- PWA verifier/test: [streamlit_app/test_pwa_setup.py](streamlit_app/test_pwa_setup.py) — see [`validate_manifest`](streamlit_app/test_pwa_setup.py), [`check_icons`](streamlit_app/test_pwa_setup.py) and [`main`](streamlit_app/test_pwa_setup.py)  
- ML model notes: [models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md](models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md)  
- Face detector: [models/haarcascade_frontalface_default.xml](models/haarcascade_frontalface_default.xml)  
- Requirements: [requirements.txt](requirements.txt)  
- License: [License](License)

## Features
- Real-time eye aspect ratio (EAR) monitoring and drowsiness alerting
- Streamlit web interface for webcam-based monitoring ([streamlit_app/streamlit_app.py](streamlit_app/streamlit_app.py))
- Optional PWA support for installable/offline usage ([streamlit_app/streamlit_app_pwa.py](streamlit_app/streamlit_app_pwa.py))
- Icon generation helper ([streamlit_app/generate_icons.py](streamlit_app/generate_icons.py)) and icon set in [streamlit_app/icons/](streamlit_app/icons/)
- Pre-trained transfer-learning model described in [models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md](models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md)

## Quickstart
1. Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r [requirements.txt](requirements.txt)

3. Run the standard app:
   streamlit run [streamlit_app/streamlit_app.py](streamlit_app/streamlit_app.py)

   Or run the PWA-enabled app:
   streamlit run [streamlit_app/streamlit_app_pwa.py](streamlit_app/streamlit_app_pwa.py)

## PWA Setup & Testing
- Ensure manifest and service worker are present: [streamlit_app/manifest.json](streamlit_app/manifest.json), [streamlit_app/sw.js](streamlit_app/sw.js)  
- Generate missing icons:
  python3 [streamlit_app/generate_icons.py](streamlit_app/generate_icons.py) — uses [`create_icon`](streamlit_app/generate_icons.py) / [`generate_all_icons`](streamlit_app/generate_icons.py) and writes to [streamlit_app/icons/](streamlit_app/icons/)

- Validate PWA files with the test script:
  python3 [streamlit_app/test_pwa_setup.py](streamlit_app/test_pwa_setup.py) — uses [`validate_manifest`](streamlit_app/test_pwa_setup.py), [`check_icons`](streamlit_app/test_pwa_setup.py) and [`main`](streamlit_app/test_pwa_setup.py)

Note: PWA install flow requires HTTPS in production (see comments in [streamlit_app/test_pwa_setup.py](streamlit_app/test_pwa_setup.py) and [streamlit_app/streamlit_app_pwa.py](streamlit_app/streamlit_app_pwa.py)).

## Models & Training
Documentation for the ResNet50 transfer-learning model and dataset details are in:
[models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md](models/ML_Models/ResNET+CNN (Tranfer Learning)/readme.md)

Face detection uses Haar cascade at:
[models/haarcascade_frontalface_default.xml](models/haarcascade_frontalface_default.xml)

## Utilities and Scripts
- [latest.py](latest.py)
- [Sesssion_analysis.py](Sesssion_analysis.py)

## Contributing
1. Fork the repo
2. Create a branch, implement changes, commit
3. Open a Pull Request

See the project board and roadmap referenced in the main [README.md](README.md) for active tasks.

## Support & Issues
Open issues on the repository to report bugs or request features.

## License
See [License](License)
