# IVF Journey Tracker - Extension Plan

## Features to ADD:

### 1. Dashboard Charts Enhancement
- [x] Embed wellness charts into patient_dashboard.html
- [x] Show mood, stress, sleep trends directly on dashboard
- [x] Add quick stats cards

### 2. Reminder System
- [x] Add medication reminder form (patients can add their own reminders)
- [x] Add reminder alerts/notifications on dashboard
- [x] Create API routes for reminders
- [x] Add route: /my_reminders

### 3. Patient IVF Tracking
- [x] Allow patients to view their IVF cycles
- [x] Add patient-side IVF cycle creation
- [x] View medication history
- [x] Add route: /my_treatment

### 4. Doctor-Patient Messaging
- [x] Add messaging model (Message)
- [x] Create message form in patient dashboard
- [x] Create messaging interface
- [x] Add route: /messages

### 5. AI Prediction Labeling
- [x] Mark AI features as "Research / Demo model" in ivf_predictor.html
- [x] Add disclaimer labels

### 6. Mindfulness Page
- [x] Create new mindfulness/meditation page
- [x] Add guided meditation content
- [x] Add route: /mindfulness

## Implementation Order:
1. Dashboard Charts Enhancement - COMPLETED
2. Reminder System (forms + display) - COMPLETED
3. Patient IVF Tracking - COMPLETED
4. Doctor-Patient Messaging - COMPLETED
5. AI Prediction Labeling - COMPLETED
6. Mindfulness Page - COMPLETED

## Files Created:
- templates/mindfulness.html
- templates/my_reminders.html
- templates/my_treatment.html
- templates/messages.html

## Files Modified:
- models.py (added Message model)
- main.py (added new routes)
- templates/patient_dashboard.html (added charts, feature links)
- templates/ivf_predictor.html (added demo labels)
- templates/base.html (added navigation for new pages)
