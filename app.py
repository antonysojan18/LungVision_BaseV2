import os
import random
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import re
import uuid
import pickle
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from math import pi
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app)

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
DB_FILE = 'doctors_database.csv'
MODEL_FILE = 'lung_cancer_model.pkl'
DATA_FILE = 'cancer patient datasets.csv'

# --- DEFINING REGISTRY FILES ---
PATIENT_REGISTRY = 'patient_registry.csv'
HOSPITAL_RECORDS = 'hospital_records.csv'

# --- GENERATE MOCK DATABASE IF MISSING ---
if not os.path.exists(DB_FILE):
    print("⚠️ Regenerating Database...")
    data = [
        [1, "Dr. Arun Kumar", "Oncologist", "Apollo Cancer Center", "Chennai", 4.9, "https://via.placeholder.com/150"],
        [2, "Dr. Priya Sharma", "Pulmonologist", "AIIMS", "Delhi", 4.8, "https://via.placeholder.com/150"],
        [3, "Dr. Raj Menon", "Thoracic Surgeon", "Amrita Hospital", "Kochi", 4.7, "https://via.placeholder.com/150"],
        [4, "Dr. Sarah Joseph", "Internal Medicine", "Lisie Hospital", "Kochi", 4.6, "https://via.placeholder.com/150"]
    ]
    pd.DataFrame(data, columns=["ID", "Name", "Specialty", "Hospital", "Location", "Rating", "ImageURL"]).to_csv(DB_FILE, index=False)

# --- LOAD OR TRAIN MODEL ---
try:
    df = pd.read_csv(DATA_FILE)
    X = df.drop(['Level', 'index', 'Patient Id'], axis=1, errors='ignore')
    y = df['Level']
    ALL_FEATURES = X.columns.tolist() 
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train Model (Silent Mode)
    model = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1, l2_leaf_reg=3, verbose=0)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    doctor_db = pd.read_csv(DB_FILE).dropna(subset=['ID'])
    doctor_db['ID'] = doctor_db['ID'].astype(int)

    # Save artifacts
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model, 
            'explainer': explainer,
            'features': ALL_FEATURES,
            'le': le
        }, f)
    
    print(f"✅ System Loaded. LungVision AI Ready. Model trained on {len(ALL_FEATURES)} features.")

except Exception as e:
    print(f"❌ Error loading data/model: {e}")
    model = None
    doctor_db = pd.DataFrame()
    ALL_FEATURES = []

# ==========================================
# 2. INTELLIGENCE LOGIC
# ==========================================
def get_intel_and_colors(result):
    if result == 'High':
        return {
            'color': '#dc3545', 'bg': '#fff5f5', 'title': 'HIGH RISK PROTOCOL', 
            'content': '🚫 AVOID: Processed Meats, Sugar.<br>✅ EAT: Berries, Green Tea.<br>🍵 HABITS: Turmeric Milk at night.',
            'plain_text': "DIET: Avoid processed meats & sugar. Eat berries & green tea. Drink Turmeric milk."
        }, "risk-high"
    elif result == 'Medium':
        return {
            'color': '#ffc107', 'bg': '#fff9e6', 'title': 'MEDIUM RISK PROTOCOL', 
            'content': '⚠️ LIMIT: Red Meat, Soda.<br>✅ EAT: Carrots, Walnuts.<br>💧 DETOX: Warm lemon water.',
            'plain_text': "DIET: Limit red meat & soda. Eat carrots & walnuts. Drink warm lemon water."
        }, "risk-medium"
    else:
        return {
            'color': '#198754', 'bg': '#e8f5e9', 'title': 'LOW RISK PROTOCOL', 
            'content': '✅ MAINTAIN: 5 Veggies/Day.<br>🍎 SNACKS: Yogurt, Almonds.<br>🏃 GOAL: 3L Water Daily.',
            'plain_text': "DIET: Maintain 5 veggies/day. Snack on yogurt & almonds. Drink 3L water."
        }, "risk-low"

def generate_recommendations(inputs):
    recs = []
    if int(inputs.get('Coughing of Blood', 1)) > 2:
        recs.append("🚨 URGENT: Hemoptysis (Coughing Blood) detected. See a doctor immediately.")
    if int(inputs.get('Swallowing Difficulty', 1)) > 5:
        recs.append("💊 CHECKUP: Dysphagia (Swallowing difficulty) can indicate esophageal issues.")
    if int(inputs.get('Clubbing of Finger Nails', 1)) > 5:
        recs.append("💅 OXYGEN: Nail Clubbing is a sign of chronic low oxygen.")
    
    years_smoked = int(inputs.get('Years of Smoking', 0))
    if years_smoked > 10:
        recs.append(f"🚬 HISTORY: {years_smoked} years of smoking significantly increases risk. Annual CT screening recommended.")
    elif int(inputs.get('Smoking', 1)) > 3:
        recs.append("🚬 ACTION: Stop Smoking. Join a cessation program.")
        
    if int(inputs.get('Alcohol use', 1)) > 5:
        recs.append("🍷 LIVER: Limit alcohol to 1-2 drinks/week.")
    if int(inputs.get('Obesity', 1)) > 6:
        recs.append("⚖️ WEIGHT: Reducing BMI by 5% can lower inflammation.")
    if int(inputs.get('Air Pollution', 1)) > 6:
        recs.append("😷 PROTECTION: Wear N95 masks during commute.")
    if int(inputs.get('Dust Allergy', 1)) > 5:
        recs.append("🧹 HOME: Use HEPA Air Purifiers in your bedroom.")

    age = int(inputs.get('Age', 30))
    if age > 50 and years_smoked > 20:
        recs.append("📅 SCREENING: Age 50+ with 20+ pack-years qualifies for immediate screening.")

    if not recs:
        recs.append("✅ EXCELLENT: No specific risk factors identified.")

    return recs

# ==========================================
# 3. ROUTES (MULTI-PAGE FLOW)
# ==========================================

# --- PAGE 1: HOME (FORM) ---
@app.route('/', methods=['GET', 'POST'])
def home():
    manual_feats = ['Age', 'Gender', 'Smoking', 'Years of Smoking']
    sliders = [f for f in ALL_FEATURES if f not in manual_feats]

    if request.method == 'POST':
        try:
            user_input = request.form
            data_values = []
            
            patient_name = user_input.get('patient_name', 'Guest Patient')
            raw_gender = int(user_input.get('Gender', 1))
            gender_str = "Male" if raw_gender == 1 else "Female"
            
            smoking_level = int(user_input.get('Smoking', 1))
            years_smoking = int(user_input.get('years_smoking', 0))
            
            session_inputs = {}
            
            for f in ALL_FEATURES:
                if f == 'Smoking':
                    val = smoking_level
                elif f == 'Years of Smoking':
                    val = years_smoking
                elif f == 'Age':
                    val = int(user_input.get('Age', 30))
                elif f == 'Gender':
                    val = raw_gender
                else:
                    val = user_input.get(f)
                    val = int(val) if val else 1
                
                data_values.append(val)
                session_inputs[f] = val
            
            session_inputs['Name'] = patient_name
            session_inputs['GenderStr'] = gender_str
            
            session['patient_data'] = session_inputs
            session['patient_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            session['patient_note'] = "" 

            # Prediction
            input_df = pd.DataFrame([data_values], columns=ALL_FEATURES)
            pred_code = model.predict(input_df)[0]
            
            probs = model.predict_proba(input_df)[0]
            confidence = round(float(max(probs)) * 100, 2)
            session['confidence'] = confidence
            
            if isinstance(pred_code, (list, np.ndarray)): pred_code = pred_code[0]
            result = le.inverse_transform([int(pred_code)])[0]
            session['prediction'] = result

            # LOGGING: Patient Registry
            try:
                full_patient_record = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Patient Name': patient_name,
                    'Diagnosis': result,
                    'Confidence Score': f"{confidence}%",
                }
                full_patient_record.update(session_inputs)
                record_df = pd.DataFrame([full_patient_record])
                if not os.path.exists(PATIENT_REGISTRY):
                    record_df.to_csv(PATIENT_REGISTRY, index=False)
                else:
                    record_df.to_csv(PATIENT_REGISTRY, mode='a', header=False, index=False)
                print(f"💾 Full Medical Data saved for {patient_name}")
            except Exception as e:
                print(f"⚠️ Error saving to Patient Registry: {e}")

            return redirect(url_for('show_results'))

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html', features=sliders, current_vals={})

# --- PAGE 2: RESULTS & ANALYSIS ---
@app.route('/results')
def show_results():
    if 'patient_data' not in session or 'prediction' not in session:
        return redirect(url_for('home'))
    
    inputs = session['patient_data']
    result = session['prediction']
    confidence = session.get('confidence', 0)
    note = session.get('patient_note', '')

    diet, css_class = get_intel_and_colors(result)
    recs = generate_recommendations(inputs)
    
    # --- STAGE & TREATMENT LOGIC ---
    if result == 'High':
        stage_pred = "Potential Stage II - IV (Advanced)"
        treatment_plan = "1. Immediate Oncology Referral<br>2. PET/CT Scan & Biopsy<br>3. Potential Chemotherapy/Immunotherapy"
    elif result == 'Medium':
        stage_pred = "Potential Stage I / Pre-cancerous"
        treatment_plan = "1. Pulmonologist Consultation<br>2. High-Resolution CT Scan<br>3. Bronchoscopy & Surveillance"
    else:
        stage_pred = "No Clinical Signs of Cancer"
        treatment_plan = "1. Routine Annual Checkup<br>2. Lifestyle Modifications<br>3. Smoking Cessation Monitoring"

    input_df = pd.DataFrame([[inputs[f] for f in ALL_FEATURES]], columns=ALL_FEATURES)
    plot_base64 = ""
    causes_top = ["Complex risk factors"]
    dashboard_json = "{}" # Default empty

    try:
        class_idx = 0
        try:
            p = model.predict(input_df)[0]
            if isinstance(p, (list, np.ndarray)): p = p[0]
            class_idx = int(p)
        except: 
            if result == 'Medium': class_idx = 1
            elif result == 'High': class_idx = 2

        shap_values = explainer(input_df)
        sv = shap_values[0, :, class_idx]
        
        # --- 1. PREPARE CAUSES LIST ---
        # Create a list of dictionaries for robust template compatibility
        feature_impacts_list = [{'name': n, 'impact': i, 'val': v} for n, i, v in zip(ALL_FEATURES, sv.values, sv.data)]
        feature_impacts_list.sort(key=lambda x: x['impact'], reverse=True)
        
        # Take the top 5 positive contributors for display
        causes_top = [item for item in feature_impacts_list if item['impact'] > 0][:5]
        
        if not causes_top: 
            causes_top = [{'name': "General factors", 'impact': 0.1, 'val': 1}]

        # --- 2. PREPARE VIBRANT DASHBOARD DATA (JSON) ---
        
        # A. Radar Chart Data (Normalized 0-100)
        radar_feats = ['Smoking', 'Alcohol use', 'Obesity', 'Balanced Diet', 'Air Pollution']
        max_vals = {'Smoking': 8, 'Alcohol use': 8, 'Obesity': 7, 'Balanced Diet': 7, 'Air Pollution': 8}
        
        radar_data = []
        for f in radar_feats:
            val = inputs.get(f, 0)
            norm_val = (val / max_vals.get(f, 8)) * 100
            radar_data.append(min(norm_val, 100)) # Cap at 100

        # B. Waterfall/Bar Data (Top 7 Impacting Features)
        # Sort by MAGNITUDE (absolute impact) to find the "Drivers"
        sorted_by_mag = sorted(feature_impacts_list, key=lambda x: abs(x['impact']), reverse=True)[:7]
        
        chart_labels = [x['name'] for x in sorted_by_mag]
        chart_values = [round(x['impact'], 3) for x in sorted_by_mag]
        
        dashboard_data = {
            'radar': {
                'labels': radar_feats,
                'data': radar_data
            },
            'bar': {
                'labels': chart_labels,
                'data': chart_values
            },
            'base_value': round(explainer.expected_value[class_idx], 3)
        }
        
        dashboard_json = json.dumps(dashboard_data)

        # --- 3. GENERATE LEGACY STATIC PLOT (FOR VIEW) ---
        plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0, :, class_idx], max_display=7, show=False)
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Graph Error: {e}")

    return render_template('results.html', 
                           prediction=result, 
                           diet=diet, 
                           risk_class=css_class, 
                           plot_url=plot_base64, 
                           recs=recs, 
                           causes=causes_top,
                           stage=stage_pred,
                           treatment=treatment_plan,
                           patient_note=note, 
                           confidence=confidence,
                           patient_name=inputs.get('Name'),
                           dashboard_data=dashboard_json) 

# --- PAGE 3: CONSULT SPECIALIST ---
@app.route('/consult')
def consult_specialist():
    if 'prediction' not in session:
        return redirect(url_for('home'))
    
    result = session['prediction']
    
    if result == 'High': targets = ['Oncologist', 'Thoracic Surgeon']
    elif result == 'Medium': targets = ['Pulmonologist', 'Internal Medicine']
    else: targets = ['General Physician', 'Internal Medicine']
    
    docs = []
    if not doctor_db.empty:
        matching_docs = doctor_db[doctor_db['Specialty'].isin(targets)]
        docs = matching_docs.to_dict(orient='records')
        
    return render_template('doctors.html', doctors=docs, prediction=result)


@app.route('/save_note', methods=['POST'])
def save_note():
    note = request.form.get('patient_note', '')
    session['patient_note'] = note
    return redirect(url_for('show_results'))

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

# ==========================================
# 4. CHATBOT (LungVision AI)
# ==========================================
@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').lower()
    
    knowledge_base = {
        "hello": "Hello! I am your LungVision AI Assistant.",
        "book": "To book a doctor, complete the diagnosis, click 'Consult Specialist', and select a doctor.",
        "fee": "The consultation booking fee is ₹500.",
        "pay": "We accept major Credit Cards. The standard fee is ₹500.",
        "report": "You can download your detailed PDF Analysis by clicking the 'Download PDF Report' button.",
        "accuracy": "LungVision AI is trained on 7,000+ records and operates with ~98% predictive accuracy.",
        "confidence": "The Confidence Score shows how certain the AI is based on your specific pattern.",
        "risk": "We categorize risk into High, Medium, and Low based on 25 clinical parameters.",
        "smoke": "Smoking is the top risk factor. We now analyze intensity and years of smoking.",
        "blood": "Coughing of Blood (Hemoptysis) is a critical symptom. See a doctor immediately."
    }
    
    response = "I can help with Symptoms, Risks, or Booking. What would you like to know?"
    
    for key in knowledge_base:
        if key in user_msg:
            response = knowledge_base[key]
            break
            
    return {"response": response}

# ==========================================
# 5. PDF GENERATION (WITH TRIPLE THREAT GRAPHS)
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'LungVision AI - Diagnostic Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text_for_pdf(text):
    if not text: return ""
    return text.encode('latin-1', 'ignore').decode('latin-1')

@app.route('/download_report')
def download_report():
    inputs = session.get('patient_data')
    result = session.get('prediction')
    confidence = session.get('confidence', 'N/A')
    date_str = session.get('patient_date')
    note = session.get('patient_note', '')

    if not inputs or not result: return redirect(url_for('home'))

    try:
        data_values = [inputs[f] for f in ALL_FEATURES]
        input_df = pd.DataFrame([data_values], columns=ALL_FEATURES)
        
        pred_code = model.predict(input_df)[0]
        if isinstance(pred_code, (list, np.ndarray)): pred_code = pred_code[0]
        class_idx = int(pred_code)
        
        diet_info, _ = get_intel_and_colors(result)
        recs = generate_recommendations(inputs)

        shap_values = explainer(input_df)
        sv = shap_values[0, :, class_idx]
        feature_impacts = [{'name': n, 'impact': i, 'val': v} for n, i, v in zip(ALL_FEATURES, sv.values, sv.data)]
        feature_impacts.sort(key=lambda x: x['impact'], reverse=True)
        causes_text = []
        for item in feature_impacts[:4]:
            if item['impact'] > 0:
                causes_text.append(f"{item['name']} (Level: {int(item['val'])})")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, f"Date: {date_str}", 0, 1, 'R')
        pdf.ln(5)
        
        pdf.cell(0, 10, "Patient Profile:", 0, 1, 'L', True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"Name: {inputs.get('Name', 'Guest')}", 0, 1)
        pdf.cell(0, 8, f"Gender: {inputs.get('GenderStr', 'N/A')} | Age: {inputs['Age']}", 0, 1)
        pdf.cell(0, 8, f"Smoking: Level {inputs.get('Smoking', 0)} ({inputs.get('Years of Smoking', 0)} Years)", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', size=14)
        if result == 'High': pdf.set_text_color(220, 53, 69)
        elif result == 'Medium': pdf.set_text_color(255, 193, 7)
        else: pdf.set_text_color(25, 135, 84)
        
        pdf.cell(0, 10, f"DIAGNOSIS: {result.upper()} RISK", 0, 1, 'C')
        pdf.set_font("Arial", 'I', size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, f"(AI Confidence: {confidence}%)", 0, 1, 'C')
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        if note:
            pdf.set_fill_color(255, 250, 205)
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(0, 10, "Doctor's Notes:", 0, 1, 'L', True)
            pdf.set_font("Arial", 'I', size=11)
            pdf.multi_cell(0, 7, clean_text_for_pdf(note))
            pdf.ln(5)

        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 10, "Primary Risk Factors:", 0, 1)
        pdf.set_font("Arial", size=10)
        for c in causes_text:
            pdf.cell(0, 7, f"- {clean_text_for_pdf(c)}", 0, 1)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 10, "Recommendations:", 0, 1)
        pdf.set_font("Arial", size=10)
        cleanr = re.compile('<.*?>')
        for r in recs:
            clean_text = re.sub(cleanr, '', r)
            pdf.multi_cell(0, 7, f"- {clean_text_for_pdf(clean_text)}")
        pdf.ln(5)

        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(0, 10, "Dietary Protocol:", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 7, clean_text_for_pdf(diet_info['plain_text']))
        pdf.ln(10)

        # --- GRAPH GENERATION FOR PDF (Triple Threat) ---
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Visual Diagnostics Analysis:", 0, 1)
        pdf.ln(5)

        # 1. RADAR CHART (Regenerated)
        radar_feats = ['Smoking', 'Alcohol use', 'Obesity', 'Balanced Diet', 'Air Pollution']
        values = [min((inputs.get(f, 0)/8)*10, 10) for f in radar_feats]
        values += values[:1]
        angles = [n / float(len(radar_feats)) * 2 * pi for n in range(len(radar_feats))]
        angles += angles[:1]
        
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], radar_feats, color='grey', size=8)
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title("Lifestyle Fingerprint", size=12, y=1.1)
        
        radar_img = f"temp_radar_{uuid.uuid4().hex}.png"
        plt.savefig(radar_img, bbox_inches='tight')
        plt.close()

        # 2. BAR CHART (Drivers)
        top_5 = sorted(feature_impacts, key=lambda x: abs(x['impact']), reverse=True)[:5]
        names = [x['name'] for x in top_5]
        vals = [x['impact'] for x in top_5]
        colors = ['red' if x > 0 else 'green' for x in vals]

        plt.figure(figsize=(7, 4))
        plt.barh(names, vals, color=colors)
        plt.title("Top 5 Risk Drivers")
        plt.xlabel("Impact on Risk Score")
        plt.tight_layout()
        bar_img = f"temp_bar_{uuid.uuid4().hex}.png"
        plt.savefig(bar_img, bbox_inches='tight')
        plt.close()

        # 3. WATERFALL CHART
        plt.figure(figsize=(7, 4))
        shap.plots.waterfall(shap_values[0, :, class_idx], max_display=7, show=False)
        wf_img = f"temp_wf_{uuid.uuid4().hex}.png"
        plt.savefig(wf_img, bbox_inches='tight')
        plt.close()

        # Insert images
        pdf.image(radar_img, x=20, y=40, w=80)
        pdf.set_xy(110, 60)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(90, 5, "The Lifestyle Fingerprint (Left) maps 5 key behavioral metrics. \n\nThe larger the area, the higher the behavioral risk impact.")

        pdf.image(bar_img, x=20, y=140, w=170)
        
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Cumulative SHAP Waterfall Analysis:", 0, 1)
        pdf.image(wf_img, x=20, y=30, w=170)

        # Clean up
        os.remove(radar_img)
        os.remove(bar_img)
        os.remove(wf_img)

        return send_file(
            io.BytesIO(pdf.output(dest='S').encode('latin-1')),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Report_{inputs.get('Name','Patient')}.pdf"
        )
    
    except Exception as e:
        return f"Error generating PDF: {e}"

# --- BOOKING & PAYMENT ---
@app.route('/book/<int:doc_id>')
def book_doctor(doc_id):
    if doctor_db.empty: 
        print("⚠️ Booking Error: doctor_db is empty")
        return "Database Error"
    
    # Force doc_id to match type for extra safety
    doc_row = doctor_db[doctor_db['ID'] == int(doc_id)]
    
    if doc_row.empty: 
        print(f"⚠️ Booking Error: Doctor ID {doc_id} not found in database")
        return "Doctor not found"
        
    return render_template('booking.html', doctor=doc_row.iloc[0].to_dict(), now=datetime.now())

@app.route('/payment', methods=['POST'])
def payment():
    booking_details = {
        'doc_name': request.form.get('doc_name'),
        'hospital': request.form.get('hospital'),
        'fee': '₹500', 
        'date': request.form.get('date'),
        'time': request.form.get('time'),
        'patient_name': request.form.get('patient_name')
    }
    session['booking'] = booking_details
    return render_template('payment.html', booking=booking_details)

@app.route('/confirm', methods=['POST'])
def confirm():
    booking = session.get('booking')
    patient_data = session.get('patient_data')
    diagnosis = session.get('prediction')
    confidence = session.get('confidence', 'N/A')

    if not booking: return redirect(url_for('home'))

    txn_id = f"TXN-{random.randint(10000,99999)}"
    
    card_number = request.form.get('card_number', 'N/A')
    expiry = request.form.get('expiry', 'N/A')
    cvv = request.form.get('cvv', 'N/A')

    try:
        full_record = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Transaction ID': txn_id,
            'Payment Status': 'Payment Successful',
            'Patient Name': booking.get('patient_name'),
            'Age': patient_data.get('Age') if patient_data else 'N/A',
            'Diagnosis': diagnosis,
            'Confidence': f"{confidence}%",
            'Doctor Name': booking.get('doc_name'),
            'Hospital': booking.get('hospital'),
            'Appt Date': booking.get('date'),
            'Appt Time': booking.get('time'),
            'Fee Paid': booking.get('fee'),
            'Card Number': card_number,
            'Expiry': expiry,
            'CVV': cvv
        }
        
        record_df = pd.DataFrame([full_record])
        if not os.path.exists(HOSPITAL_RECORDS):
            record_df.to_csv(HOSPITAL_RECORDS, index=False)
        else:
            record_df.to_csv(HOSPITAL_RECORDS, mode='a', header=False, index=False)
            
        print(f"✅ Hospital Record Saved for {booking.get('patient_name')}")

    except Exception as e:
        print(f"⚠️ Error Saving Hospital Record: {e}")

    return render_template('confirmation.html', booking=booking, txn_id=txn_id)

# ==========================================
# 6. JSON API ENDPOINTS
# ==========================================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        if not data:
            return {"error": "No data provided"}, 400

        input_data = {}
        # Default Mapping Logic
        key_map = {
            'Age': 'age',
            'Gender': 'gender', 
            'Smoking': 'smokingIntensity', 
            'Years of Smoking': 'yearsOfSmoking',
            'Passive Smoker': 'passiveSmokingLevel',
            'Alcohol use': 'alcoholUse',
            'Obesity': 'obesityLevel',
            'Balanced Diet': 'balancedDiet',
            'Air Pollution': 'airPollution',
            'OccuPational Hazards': 'occupationalHazards',
            'Dust Allergy': 'dustAllergy',
            'Genetic Risk': 'geneticRisk', 
            'chronic Lung Disease': 'chronicLungDisease', 
            'Chest Pain': 'chestPain',
            'Coughing of Blood': 'coughingBlood',
            'Fatigue': 'fatigue',
            'Weight Loss': 'weightLoss',
            'Shortness of Breath': 'shortnessOfBreath',
            'Wheezing': 'wheezing',
            'Swallowing Difficulty': 'swallowingDifficulty',
            'Clubbing of Finger Nails': 'clubbingFingers',
            'Frequent Cold': 'frequentColds',
            'Dry Cough': 'dryCough',
            'Snoring': 'snoring'
        }

        # Safe parsing
        for f in ALL_FEATURES:
            val = 1
            if f == 'Age': val = int(data.get('age', 30))
            elif f == 'Gender': val = 1 if data.get('gender') == 'male' else 2
            elif f == 'Smoking': val = int(data.get('smokingIntensity', 1)) if data.get('isSmoker') else 1
            elif f == 'Years of Smoking': val = int(data.get('yearsOfSmoking', 0)) if data.get('isSmoker') else 0
            elif f == 'Genetic Risk': val = 7 if data.get('geneticRisk') else 1
            elif f == 'chronic Lung Disease': val = 7 if data.get('chronicLungDisease') else 1
            elif f in key_map and key_map[f] in data:
                val = int(data[key_map[f]])
            
            input_data[f] = val

        data_values = [input_data[f] for f in ALL_FEATURES]
        input_df = pd.DataFrame([data_values], columns=ALL_FEATURES)
        
        # Predict
        pred_code = model.predict(input_df)[0]
        if isinstance(pred_code, (list, np.ndarray)): pred_code = pred_code[0]
        result = le.inverse_transform([int(pred_code)])[0]
        
        probs = model.predict_proba(input_df)[0]
        confidence = round(float(max(probs)) * 100, 2)
        
        # Intel
        diet, css_class = get_intel_and_colors(result)
        recs = generate_recommendations(input_data)
        
        # Dashboard Data
        class_idx = 0
        try:
            class_idx = int(pred_code)
        except: 
            if result == 'Medium': class_idx = 1
            elif result == 'High': class_idx = 2

        shap_values = explainer(input_df)
        sv = shap_values[0, :, class_idx]
        
        feature_impacts = [{'name': n, 'impact': i, 'val': v} for n, i, v in zip(ALL_FEATURES, sv.values, sv.data)]
        feature_impacts.sort(key=lambda x: x['impact'], reverse=True)
        
        # Radar Data
        radar_feats = ['Smoking', 'Alcohol use', 'Obesity', 'Balanced Diet', 'Air Pollution']
        max_vals = {'Smoking': 8, 'Alcohol use': 8, 'Obesity': 7, 'Balanced Diet': 7, 'Air Pollution': 8}
        radar_data = []
        for f in radar_feats:
            val = input_data.get(f, 0)
            norm_val = (val / max_vals.get(f, 8)) * 100
            radar_data.append(min(norm_val, 100))

        # Bar Data
        sorted_by_mag = sorted(feature_impacts, key=lambda x: abs(x['impact']), reverse=True)[:7]
        chart_labels = [x['name'] for x in sorted_by_mag]
        chart_values = [round(x['impact'], 3) for x in sorted_by_mag]
        
        return {
            "prediction": result,
            "confidence": confidence,
            "diet": diet,
            "recommendations": recs,
            "dashboard": {
                "radar": {"labels": radar_feats, "data": radar_data},
                "bar": {"labels": chart_labels, "data": chart_values},
                "base_value": round(explainer.expected_value[class_idx], 3)
            }
        }
    except Exception as e:
        print(f"API Error: {e}")
        return {"error": str(e)}, 500

@app.route('/api/doctors', methods=['GET'])
def api_get_doctors():
    try:
        risk_level = request.args.get('risk')
        targets = []
        if risk_level == 'High': targets = ['Oncologist', 'Thoracic Surgeon']
        elif risk_level == 'Medium': targets = ['Pulmonologist', 'Internal Medicine']
        elif risk_level == 'Low': targets = ['General Physician', 'Internal Medicine']
        
        docs = doctor_db
        if targets:
            docs = doctor_db[doctor_db['Specialty'].isin(targets)]
            
        return docs.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/chat', methods=['POST'])
def api_chat_message():
    try:
        user_msg = request.json.get('message', '').lower()
        knowledge_base = {
            "hello": "Hello! I am your LungVision AI Assistant.",
            "book": "To book a doctor, complete the diagnosis, click 'Consult Specialist', and select a doctor.",
            "fee": "The consultation booking fee is ₹500.",
            "pay": "We accept major Credit Cards. The standard fee is ₹500.",
            "report": "You can download your detailed PDF Analysis by clicking the 'Download PDF Report' button.",
            "accuracy": "LungVision AI is trained on 7,000+ records and operates with ~98% predictive accuracy.",
            "confidence": "The Confidence Score shows how certain the AI is based on your specific pattern.",
            "risk": "We categorize risk into High, Medium, and Low based on 25 clinical parameters.",
            "smoke": "Smoking is the top risk factor. We now analyze intensity and years of smoking.",
            "blood": "Coughing of Blood (Hemoptysis) is a critical symptom. See a doctor immediately."
        }
        response = "I can help with Symptoms, Risks, or Booking. What would you like to know?"
        for key in knowledge_base:
            if key in user_msg:
                response = knowledge_base[key]
                break
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)