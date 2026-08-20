# IVF Journey Tracker - Complete Project Prompt

---

## Project Overview

Build an **AI-Powered IVF Journey Tracking System** using Flask, Machine Learning, and Artificial Intelligence to help couples throughout their fertility treatment journey.

---

## Core Problem Statement

Infertility affects 1 in 8 couples globally, with IVF success rates at just 30-40%. This project presents an AI-powered web platform that helps couples navigate their fertility treatment using:
- Machine Learning for success prediction (89.8% accuracy)
- OCR document analysis
- NLP-powered chatbot
- Real-time wellness monitoring
- Comprehensive clinic database (1,500+ clinics)

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Flask + SQLAlchemy |
| **Frontend** | Bootstrap 5 + Chart.js |
| **ML Models** | XGBoost, LightGBM, BNN (Bayesian Neural Networks) |
| **AI Services** | Gemini/OpenAI API |
| **OCR** | Tesseract, pdfminer |
| **Database** | SQLite (default), PostgreSQL (production) |

---

## Key Features to Implement

### 1. AI/ML Features
- **IVF Success Prediction** with 89.8% accuracy using Genetic Algorithm + XGBoost + LightGBM
- **Emotion Detection** from text input using NLP
- **Mood Prediction** using ML models
- **Feature Selection** using Genetic Algorithm
- **OCR Document Analysis** - Extract AMH, FSH, LH, AFC from medical reports

### 2. Patient Features
- Dashboard with treatment progress
- Wellness tracking (mood, stress, sleep, exercise, nutrition)
- Document upload with OCR extraction
- IVF Success Predictor with detailed analysis
- Medication reminders (smart notifications)
- AI Chatbot (24/7 support using Gemini)
- Nutrition guidance & yoga routines
- Community forum
- Treatment cycle management
- Doctor messaging

### 3. Doctor Features
- Patient management dashboard
- Patient list with search/filter
- Cycle notes (add/view)
- Appointment scheduling
- Patient messaging system
- View patient wellness logs

### 4. Admin Features
- User management (CRUD)
- Clinic management (CRUD)
- Analytics dashboard
- Model retraining
- View audit logs

---

## Database Models Required

```
python
# User - Base user model with role-based access
User (id, username, email, password_hash, first_name, last_name, 
      user_type, phone, date_of_birth, clinic_id, created_at)

# PatientData - Extended patient information
PatientData (user_id, age, height, weight, bmi, amh_level, fsh_level, 
             diagnosis, medical_history, medications, allergies, 
             lifestyle_factors, previous_pregnancies, previous_ivf_cycles, 
             partner_age, partner_diagnosis)

# IVFCycle - IVF treatment cycles
IVFCycle (patient_id, protocol, start_date, status, outcome, 
         patient_notes, created_at)

# WellnessLog - Daily wellness tracking
WellnessLog (user_id, date, mood_rating, mood_notes, stress_level, 
            stress_factors, sleep_hours, sleep_quality, exercise_minutes, 
            meditation_minutes, yoga_practiced, water_intake, nutrition_score,
            supplements_taken, detected_emotion, meal_breakfast, meal_lunch,
            meal_dinner, meal_snacks)

# MedicationReminder - Medication tracking
MedicationReminder (user_id, medication_name, dosage, frequency, 
                   time_of_day, start_date, end_date, is_active)

# MedicalDocument - Document storage with OCR
MedicalDocument (user_id, filename, original_filename, file_type, 
                file_size, description, extracted_text, uploaded_at)

# Prediction - AI prediction results
Prediction (user_id, success_probability, protocol_recommendation, 
           llm_analysis, prediction_date)

# CycleNote - Doctor notes on patient cycles
CycleNote (cycle_id, doctor_id, note_content, created_at)

# Clinic - IVF clinic database
Clinic (name, address, city, state, zip_code, phone, website, 
       latitude, longitude, description, clinic_type)

# Message - Patient-Doctor messaging
Message (sender_id, receiver_id, subject, content, is_read, created_at)

# AuditLog - Security audit trail
AuditLog (user_id, action, details, ip_address, created_at)

# ForumPost - Community forum posts
ForumPost (title, content, user_id, created_at)

# MedicalActivity - Track medical activities/appointments
MedicalActivity (patient_id, activity_type, activity_name, 
                performed_date, notes)
```

---

## Routes/Endpoints Required

### Authentication
- `GET /` - Landing page (redirects based on user_type)
- `GET/POST /login` - User login
- `GET/POST /register` - Patient registration
- `GET/POST /register/doctor` - Doctor registration
- `GET /logout` - User logout
- `GET/POST /reset_password` - Password reset
- `GET/POST /reset_password/<token>` - Reset with token

### Patient Routes
- `GET /patient_dashboard` - Patient dashboard
- `GET /ivf_predictor` - IVF success prediction page
- `GET/POST /wellness` - Wellness tracking
- `GET /nutrition` - Nutrition & yoga guidance
- `GET /chatbot` - AI chatbot interface
- `GET/POST /my_documents` - Document management
- `GET /my_treatment` - Treatment cycle management
- `GET/POST /my_reminders` - Medication reminders
- `GET /messages` - Doctor messaging
- `GET/POST /update_profile` - Profile management
- `GET /find_clinic` - Search clinics

### Doctor Routes
- `GET /doctor_dashboard` - Doctor dashboard
- `GET /patient/<id>` - View patient details
- `GET/POST /edit_patient/<id>` - Edit patient info
- `GET /my_notes` - View own notes
- `GET /doctor_messages` - Patient messages
- `GET/POST /doctor_messages/<patient_id>` - View conversation
- `GET/POST /add_cycle_note/<cycle_id>` - Add note to cycle
- `GET/POST /create_cycle` - Create new IVF cycle
- `GET/POST /schedule_appointment` - Schedule appointment

### Admin Routes
- `GET /admin` - Admin dashboard
- `GET /admin/clinics` - Manage clinics
- `GET/POST /admin/clinic/add` - Add clinic
- `GET/POST /admin/clinic/edit/<id>` - Edit clinic
- `POST /admin/clinic/delete/<id>` - Delete clinic
- `GET /admin/users` - Manage users
- `GET/POST /admin/user/edit/<id>` - Edit user
- `POST /admin/user/delete/<id>` - Delete user
- `POST /admin/retrain_model` - Retrain ML model

### API Endpoints
- `POST /api/chat` - Chat with AI (Gemini)
- `POST /api/chat/tts` - Text-to-speech
- `POST /predict_ivf` - IVF prediction API
- `POST /predict_mood` - Mood prediction
- `POST /analyze_emotion` - Emotion detection
- `GET /api/wellness_data` - Wellness data for charts
- `POST /upload_document` - Document upload with OCR
- `POST /api/predict_ivf_ml` - ML prediction endpoint

### Static Pages
- `GET /faq` - FAQ page
- `GET /privacy_policy` - Privacy policy
- `GET /business_model` - Business model
- `GET /roadmap` - Project roadmap
- `GET /data_security` - Data security info
- `GET /forum` - Community forum
- `GET/POST /forum/new` - Create forum post
- `GET /forum/post/<id>` - View post
- `GET /mindfulness` - Mindfulness exercises

---

## ML Model Specifications

### Features for Prediction (10 Critical Features)
1. Female Age
2. BMI
3. AMH Level
4. FSH Level
5. Previous IVF Attempts
6. Stress Level
7. Sleep Hours
8. Exercise Minutes
9. Endometrial Thickness
10. Sperm Parameters

### Model Training Pipeline
- Use Genetic Algorithm for feature selection
- Train XGBoost, LightGBM, and BNN models
- Achieve 89.8% prediction accuracy
- Save models as `.pkl` files in `models/` directory

### Model Files Required
```
models/
├── ivf_success_model.pkl      # Main IVF prediction model
├── mood_trend_model.pkl       # Mood prediction model
├── emotion_model.pkl          # Emotion detection model
├── emotion_vectorizer.pkl     # Text vectorizer for emotion
└── ivf_model_metadata.json    # Model metadata
```

---

## File Structure

```
ChoreCompanion/
├── main.py                    # Main Flask application (all routes)
├── database.py                # SQLAlchemy database initialization
├── models.py                  # Database models (User, PatientData, etc.)
├── openai_service.py          # Gemini/OpenAI API services
├── prediction_service.py      # ML prediction service
├── predict.py                 # Prediction utilities
├── train_models.py            # Model training script
├── requirements.txt           # Python dependencies
├── ml_models/
│   ├── __init__.py
│   ├── model_factory.py       # Model factory pattern
│   ├── model_training.py      # Training pipeline
│   ├── preprocessing.py       # Data preprocessing
│   ├── xgboost_models.py      # XGBoost implementations
│   ├── lightgbm_models.py    # LightGBM implementations
│   ├── bnn_models.py          # BNN implementations
│   ├── explanation.py        # SHAP explanations
│   └── saved_models/         # Saved model weights
├── models/                    # Trained .pkl models
├── templates/                 # HTML templates (40+ files)
├── static/
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # Static images
├── uploads/                   # User uploaded documents
├── instance/                  # SQLite database
└── tests/                     # Unit tests
```

---

## Default Login Credentials

| User Type | Username | Password |
|-----------|----------|----------|
| Admin | admin | admin123 |
| Doctor | doctor | doctor123 |
| Patient | (Register new) | (Register new) |

---

## Sample Clinic Data (Pre-seeded)

### USA Clinics
- Hope Fertility Center (New York, NY)
- Pinnacle IVF (Los Angeles, CA)
- Aspire Fertility Institute (Chicago, IL)
- Genesis Fertility Clinic (Houston, TX)

### India - Bangalore
- Nova IVF Fertility
- Oasis Fertility
- Iswarya Fertility Centre
- Gaudium IVF Centre
- Indira IVF

### India - Mumbai
- Bloom IVF Center
- ART Fertility Clinics
- Bavishi Fertility Institute
- Progenesis IVF

### India - Delhi
- International Fertility Centre
- Medicover Fertility
- Aakash Healthcare

### India - Other Cities
- Chennai: Prashanth Fertility, Jananam Fertility
- Hyderabad: KIMS Cuddles, Srujana Fertility
- Mysore: Gunasheela Fertility Centre
- Mangalore: Manipal Fertility

---

## Key Implementation Highlights

### 1. OCR Document Analysis
- Extract AMH, FSH, LH, AFC from medical reports
- Support PDF, Image (JPG, PNG), DOCX formats
- Auto-fill patient profile with extracted data
- Generate AI insights from extracted parameters

### 2. AI Chatbot (24/7)
- Gemini API integration
- Context-aware responses
- TTS audio generation
- Wellness tips and IVF guidance

### 3. Wellness Tracking
- Daily mood, stress, sleep, exercise logging
- Emotion detection from mood notes (NLP)
- Interactive charts (Chart.js)
- Nutrition tracking with AI analysis

### 4. IVF Success Prediction
- ML model with 89.8% accuracy
- Personalized protocol recommendations
- Embryo quality scoring
- Factor analysis (what affects success)

### 5. Medication Reminders
- Automated reminders
- Start/end date tracking
- Active/inactive toggle
- Dosage & frequency management

### 6. Clinic Database
- 1,500+ IVF clinics
- Search by name, city, state
- Filter by clinic type
- Map integration (lat/long)

### 7. Security Features
- Password hashing (werkzeug)
- Session management
- Audit logging (HIPAA compliance)
- Role-based access control
- SQL injection prevention

---

## Running the Application

```
bash
# Install dependencies
pip install -r requirements.txt

# Initialize database (automatic on first run)
python main.py

# Access at http://localhost:5000
# Or http://127.0.0.1:5000
```

---

## Configuration

### Environment Variables (Optional)
```
SESSION_SECRET=your-secret-key
DATABASE_URL=sqlite:///ivf_tracker.db
OPENAI_API_KEY=your-api-key
GEMINI_API_KEY=your-gemini-key
```

### Key Settings (main.py)
- Secret key for sessions
- Database URI
- Upload folder (16MB max)
- Allowed file extensions

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Prediction Accuracy | 89.8% |
| Document Processing | Automated OCR |
| Chatbot Availability | 24/7 |
| Clinic Database | 1,500+ |
| User Satisfaction | High |

---

## Future Scope

- Mobile apps (iOS/Android)
- Wearable device integration
- Telemedicine video calls
- IUI/ICSI/Surrogacy support
- Multi-language support
- Insurance integration

---

*This project provides couples with data-driven decisions and 24/7 AI support, improving their chances of successful outcomes while reducing emotional and financial burden.*
