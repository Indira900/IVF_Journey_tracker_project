# TODO: Generate 5000 IVF Records and Train ML Model

## Information Gathered:
- Current model uses 8 features: age, bmi, amh, fsh, previous_ivf, stress, sleep_hours, exercise_min
- Current accuracy: 75% (from 100 records in ivf_datasets.xlsx)
- Current training in train_models.py uses ivf_datasets.xlsx
- Prediction API in main.py expects 8 features

## Plan:
- [ ] ChoreCompanion/generate_ivf_5000.py
  - [ ] Create synthetic data generator with 5000 records
  - [ ] Use 8 features to match existing model structure
  - [ ] Include realistic success rules based on medical criteria
  
- [ ] ChoreCompanion/train_ivf_model_5000.py  
  - [ ] Train RandomForest on 5000 synthetic records
  - [ ] Update metadata with new feature order
  - [ ] Expected accuracy: 85-92%

- [ ] Run training and verify model works with existing prediction API

## Dependent Files:
- main.py (no changes needed - already expects 8 features)
- train_models.py (existing - will be supplemented)
- models/ivf_success_model.pkl (will be updated)
- models/ivf_model_metadata.json (will be updated)

## Followup Steps:
- [ ] Run generate_ivf_5000.py to create dataset
- [ ] Run train_ivf_model_5000.py to train model
- [ ] Test prediction API to verify it works
- [ ] Check accuracy improvement
