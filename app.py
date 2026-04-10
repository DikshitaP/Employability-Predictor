import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ ADDED: Auto-download logic
MODEL_FILENAME = "placement_model_best.pt"

def get_model_path():
    if os.path.exists(MODEL_FILENAME):
        print("✅ Model found locally")
        return MODEL_FILENAME

    print("⬇️ Model not found. Downloading from Hugging Face...")

    model_path = hf_hub_download(
        repo_id="DikshitaP/student-placement-model",
        filename=MODEL_FILENAME,
        local_dir=".",  
        local_dir_use_symlinks=False
    )

    print("✅ Model downloaded successfully")
    return model_path

# ─── Feature lists (must match training exactly) ───────────────────────────
STUDENT_FEATURES = [
    "cgpa", "tenth_percentage", "twelfth_percentage",
    "num_technical_skills", "num_soft_skills", "num_certifications",
    "internships_done", "projects_completed", "backlogs",
    "communication_score", "aptitude_score", "hackathons_participated",
    "open_source_contributions", "research_papers", "soft_skills_rating",
]

HYBRID_TAB_COLS = STUDENT_FEATURES + [
    "skill_match_ratio", "company_min_cgpa_req", "company_internship_required",
    "company_min_projects_req", "company_min_comm_req", "company_tier_enc",
]

TIER_MAP = {
    "Tier-1 Product":     4,
    "Tier-2 Product":     3,
    "Tier-2 Core":        2,
    "PSU/Research":       2,
    "Tier-2 IT Services": 1,
}

KNOWN_COMPANIES = {
    "google":      {"tier": "Tier-1 Product",     "domain": "Product",             "min_cgpa": 8.5, "min_intern": 1, "min_proj": 3, "min_comm": 8, "skills": ["Python","Go","C++","Algorithms","Distributed Systems","System Design"], "salary_lpa": "40-80", "work_mode": "Hybrid", "location": "Bangalore/Hyderabad"},
    "microsoft":   {"tier": "Tier-1 Product",     "domain": "Product",             "min_cgpa": 8.0, "min_intern": 1, "min_proj": 3, "min_comm": 7, "skills": ["C#","Python","Java","Azure","Algorithms","OOP"], "salary_lpa": "40-70", "work_mode": "Hybrid", "location": "Hyderabad/Noida"},
    "amazon":      {"tier": "Tier-1 Product",     "domain": "E-commerce",          "min_cgpa": 7.5, "min_intern": 1, "min_proj": 2, "min_comm": 8, "skills": ["Data Structures","Algorithms","Java","System Design","AWS"], "salary_lpa": "35-65", "work_mode": "WFO", "location": "Bangalore/Hyderabad"},
    "meta":        {"tier": "Tier-1 Product",     "domain": "Product",             "min_cgpa": 8.5, "min_intern": 1, "min_proj": 3, "min_comm": 8, "skills": ["C++","Python","Distributed Systems","Algorithms"], "salary_lpa": "50-90", "work_mode": "Hybrid", "location": "Bangalore"},
    "apple":       {"tier": "Tier-1 Product",     "domain": "Product",             "min_cgpa": 8.5, "min_intern": 1, "min_proj": 3, "min_comm": 8, "skills": ["Swift","C++","Algorithms","System Design"], "salary_lpa": "45-80", "work_mode": "WFO", "location": "Hyderabad"},
    "openai":      {"tier": "Tier-1 Product",     "domain": "Artificial Intelligence","min_cgpa": 8.5, "min_intern": 1, "min_proj": 3, "min_comm": 9, "skills": ["Python","PyTorch","Linear Algebra","Transformers","Research"], "salary_lpa": "60-100", "work_mode": "Remote", "location": "Remote/San Francisco"},
    "deepmind":    {"tier": "Tier-1 Product",     "domain": "AI/ML",               "min_cgpa": 8.5, "min_intern": 1, "min_proj": 3, "min_comm": 9, "skills": ["Python","PyTorch","Linear Algebra","Calculus","Research"], "salary_lpa": "55-95", "work_mode": "Hybrid", "location": "London/Remote"},
    "nvidia":      {"tier": "Tier-1 Product",     "domain": "Hardware/Software",   "min_cgpa": 8.0, "min_intern": 1, "min_proj": 3, "min_comm": 7, "skills": ["C++","CUDA","Python","Algorithms","Parallel Computing"], "salary_lpa": "40-70", "work_mode": "WFO", "location": "Pune/Bangalore"},
    "intel":       {"tier": "Tier-1 Product",     "domain": "Hardware",            "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 7, "skills": ["Verilog","Digital Electronics","C++","VLSI"], "salary_lpa": "25-45", "work_mode": "Hybrid", "location": "Bangalore/Hyderabad"},
    "infosys":     {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 2, "min_comm": 5, "skills": ["Python","Java","SQL","OOP","Communication","Teamwork"], "salary_lpa": "3.5-6", "work_mode": "WFO/Hybrid", "location": "Pan-India"},
    "tcs":         {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Java","SQL","Python","OOP","Communication"], "salary_lpa": "3.5-7", "work_mode": "WFO/Hybrid", "location": "Pan-India"},
    "wipro":       {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Python","Basic Coding","Communication","Teamwork"], "salary_lpa": "3.5-5.5", "work_mode": "Hybrid", "location": "Pan-India"},
    "cognizant":   {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 2, "min_comm": 5, "skills": ["Python","Java","SQL","OOP","Communication","Teamwork"], "salary_lpa": "4-7", "work_mode": "Hybrid", "location": "Pan-India"},
    "accenture":   {"tier": "Tier-2 IT Services", "domain": "Consulting",          "min_cgpa": 6.5, "min_intern": 0, "min_proj": 1, "min_comm": 6, "skills": ["Java","Cloud","Communication","Problem Solving"], "salary_lpa": "4.5-8", "work_mode": "Hybrid", "location": "Pan-India"},
    "hcl":         {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Java","Python","SQL","Communication"], "salary_lpa": "3.5-5.5", "work_mode": "WFO", "location": "Pan-India"},
    "capgemini":   {"tier": "Tier-2 IT Services", "domain": "IT Services",         "min_cgpa": 6.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Java","Python","SQL","OOP"], "salary_lpa": "4-6.5", "work_mode": "Hybrid", "location": "Pan-India"},
    "flipkart":    {"tier": "Tier-2 Product",     "domain": "E-commerce",          "min_cgpa": 7.5, "min_intern": 0, "min_proj": 2, "min_comm": 7, "skills": ["Java","Python","Data Structures","Algorithms","System Design"], "salary_lpa": "20-40", "work_mode": "WFO", "location": "Bangalore"},
    "zomato":      {"tier": "Tier-2 Product",     "domain": "E-commerce",          "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["React","Node.js","MongoDB","JavaScript","System Design"], "salary_lpa": "15-30", "work_mode": "Hybrid", "location": "Gurgaon/Bangalore"},
    "swiggy":      {"tier": "Tier-2 Product",     "domain": "E-commerce",          "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["Python","Java","Microservices","React","System Design"], "salary_lpa": "15-30", "work_mode": "Hybrid", "location": "Bangalore"},
    "razorpay":    {"tier": "Tier-2 Product",     "domain": "Fintech",             "min_cgpa": 6.0, "min_intern": 0, "min_proj": 2, "min_comm": 7, "skills": ["Fullstack","JavaScript","Problem Solving","Node.js"], "salary_lpa": "18-35", "work_mode": "Hybrid", "location": "Bangalore"},
    "paytm":       {"tier": "Tier-2 Product",     "domain": "Fintech",             "min_cgpa": 6.5, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["Java","Python","SQL","Microservices"], "salary_lpa": "10-20", "work_mode": "WFO", "location": "Noida/Bangalore"},
    "ola":         {"tier": "Tier-2 Product",     "domain": "Mobility",            "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["Java","Python","Distributed Systems","Go"], "salary_lpa": "12-25", "work_mode": "WFO", "location": "Bangalore"},
    "freshworks":  {"tier": "Tier-2 Product",     "domain": "SaaS",                "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 7, "skills": ["Ruby on Rails","Python","React","SQL","System Design"], "salary_lpa": "14-28", "work_mode": "Hybrid", "location": "Chennai/Bangalore"},
    "zoho":        {"tier": "Tier-2 Product",     "domain": "SaaS",                "min_cgpa": 6.5, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["Java","C","Python","SQL","OOP"], "salary_lpa": "8-18", "work_mode": "WFO", "location": "Chennai/Hyderabad"},
    "drdo":        {"tier": "PSU/Research",       "domain": "Defence Research",    "min_cgpa": 7.0, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["C","C++","MATLAB","Embedded Systems"], "salary_lpa": "6-10", "work_mode": "WFO", "location": "Delhi/Hyderabad"},
    "isro":        {"tier": "PSU/Research",       "domain": "Space Research",      "min_cgpa": 7.5, "min_intern": 0, "min_proj": 2, "min_comm": 6, "skills": ["C","C++","MATLAB","Control Systems","Embedded Systems"], "salary_lpa": "6-12", "work_mode": "WFO", "location": "Bangalore/Thiruvananthapuram"},
    "bel":         {"tier": "PSU/Research",       "domain": "Defence Electronics", "min_cgpa": 7.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Electronics","Embedded C","VLSI","Communication Systems"], "salary_lpa": "5-9", "work_mode": "WFO", "location": "Bangalore/Pan-India"},
    "bhel":        {"tier": "PSU/Research",       "domain": "Power/Manufacturing", "min_cgpa": 7.0, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Electrical","Mechanical","CAD","PLC"], "salary_lpa": "5-9", "work_mode": "WFO", "location": "Pan-India"},
    "ongc":        {"tier": "PSU/Research",       "domain": "Oil & Gas",           "min_cgpa": 6.5, "min_intern": 0, "min_proj": 1, "min_comm": 5, "skills": ["Petroleum Engineering","Geology","Chemical Engineering"], "salary_lpa": "6-11", "work_mode": "WFO", "location": "Pan-India"},
}

# ─── Model Architecture (must match training exactly) ──────────────────────
class TabBranch(nn.Module):
    def __init__(self, in_dim, out_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout / 2),
        )
    def forward(self, x): return self.net(x)

class BertLSTMHybrid(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.bert       = BertModel.from_pretrained("bert-base-uncased")
        self.lstm       = nn.LSTM(768, 256, 2, batch_first=True, dropout=0.4)
        self.text_proj  = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.4))
        self.tab_branch = TabBranch(tab_dim, 64, 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(192, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
    def forward(self, input_ids, attention_mask, token_type_ids, tabular):
        bert_out    = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        _, (h_n, _) = self.lstm(bert_out.last_hidden_state)
        text_emb    = self.text_proj(h_n[-1])
        tab_emb     = self.tab_branch(tabular)
        return self.classifier(torch.cat([text_emb, tab_emb], dim=1)).squeeze(1)

# ─── Globals ──────────────────────────────────────────────────────────────
placement_model = None
tokenizer       = None
scaler_hybrid   = None
scaler_student  = None
emp_model       = None
THRESHOLD       = 0.53
TEMPERATURE     = 0.7862
models_loaded   = False
load_error      = None

def load_models():
    global placement_model, tokenizer, scaler_hybrid, scaler_student, emp_model
    global models_loaded, load_error, THRESHOLD, TEMPERATURE
    try:
        import sklearn
        print(f"[BOOT] Device: {DEVICE}")
        ckpt_path = os.environ.get("PLACEMENT_MODEL_PATH", get_model_path())        
        print(f"[BOOT] Loading placement model from: {ckpt_path}")
        torch.serialization.add_safe_globals([sklearn.preprocessing._data.StandardScaler])
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        placement_model = BertLSTMHybrid(tab_dim=len(HYBRID_TAB_COLS)).to(DEVICE)
        placement_model.load_state_dict(ckpt["model"])
        placement_model.eval()

        if "scaler_hybrid" in ckpt:
            scaler_hybrid  = ckpt["scaler_hybrid"]
            scaler_student = ckpt["scaler_student"]
            print("[BOOT] Scalers loaded from .pt checkpoint.")
        else:
            with open(os.environ.get("SCALER_HYBRID_PATH",  "scaler_hybrid.pkl"),  "rb") as f: scaler_hybrid  = pickle.load(f)
            with open(os.environ.get("SCALER_STUDENT_PATH", "scaler_student.pkl"), "rb") as f: scaler_student = pickle.load(f)
            print("[BOOT] Scalers loaded from .pkl files.")

        if "threshold"   in ckpt: THRESHOLD    = ckpt["threshold"]
        if "temperature" in ckpt: TEMPERATURE  = ckpt["temperature"]

        emp_path = os.environ.get("EMP_MODEL_PATH", "emp_model.pkl")
        with open(emp_path, "rb") as f: emp_model = pickle.load(f)
        print("[BOOT] Employability model loaded.")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("[BOOT] Tokenizer ready.")
        models_loaded = True
        print("[BOOT] ✅ All models loaded.")
    except Exception as e:
        load_error = str(e)
        print(f"[BOOT] ❌ Error loading models: {e}")

# ─── Helpers ───────────────────────────────────────────────────────────────
def compute_skill_match(student_skills: list, required_skills: list) -> float:
    if not required_skills:
        return 0.5
    student_lower  = {s.lower().strip() for s in student_skills}
    required_lower = [r.lower().strip() for r in required_skills]
    matched = sum(
        1 for req in required_lower
        if req in student_lower or any(req in s or s in req for s in student_lower)
    )
    return round(matched / len(required_lower), 4)

def resolve_company(company_name: str):
    if not company_name: return None
    return KNOWN_COMPANIES.get(company_name.lower().strip())

def build_profile_text(d: dict) -> str:
    tech_str = ", ".join(d["tech_skills"])
    soft_str = ", ".join(d["soft_skills"])
    cert_str = ", ".join(d["certifications"]) if d["certifications"] else "none"
    req_str  = ", ".join(d.get("company_required_skills", [])) or "general requirements"
    comm_w   = "excellent" if d["communication_score"] >= 8 else "moderate" if d["communication_score"] >= 5 else "weak"
    apt_w    = "strong" if d["aptitude_score"] >= 80 else "average" if d["aptitude_score"] >= 60 else "below-average"
    i_txt    = f"Has completed {d['internships']} internship(s)." if d["internships"] > 0 else "No prior internship experience."
    b_txt    = f"Academic record shows {d['backlogs']} active backlog(s)." if d["backlogs"] > 0 else "Academic record shows no active backlogs."
    c_txt    = f"Holds certifications including {cert_str}." if d["certifications"] else "No certifications on record."
    return (
        f"A {d['branch']} student graduating in {d['year']} with a CGPA of {d['cgpa']:.2f} out of 10. "
        f"Core technical competencies include {tech_str}. Soft skills: {soft_str}. "
        f"{i_txt} Completed {d['projects']} project(s) during the course of study. "
        f"{b_txt} {c_txt} Participated in {d['hackathons']} hackathon(s). "
        f"Demonstrates {comm_w} communication and {apt_w} aptitude (score: {d['aptitude_score']:.1f}/100). "
        f"Applied to {d.get('applied_company','the company')} ({d.get('company_domain','IT')}, {d.get('company_tier','Tier-2 IT Services')}). "
        f"Company requires: {req_str}. Skill overlap with company requirements: {d.get('skill_match_ratio', 0.5):.2f}."
    )

def get_emp_category(prob: float):
    if prob >= 0.85: return "Very High",  "#1565c0"
    if prob >= 0.70: return "High",        "#2e7d32"
    if prob >= 0.50: return "Moderate",    "#f57f17"
    if prob >= 0.30: return "Low",         "#e65100"
    return "Very Low", "#b71c1c"

def apply_guardrails(raw_placement, raw_emp, cgpa, backlogs, company_min_cgpa):
    """
    Data-calibrated post-prediction adjustments from the notebook.
    Each penalty is justified by actual placement rates in the training dataset.
    """
    adj_place = raw_placement
    adj_emp   = raw_emp
    reasons   = []

    # CGPA below company minimum → hard cap (0% actual rate → capped at 8%)
    if cgpa < company_min_cgpa:
        adj_place = min(adj_place, 0.08)
        reasons.append(f"CGPA {cgpa:.1f} is below company minimum {company_min_cgpa:.1f}")

    # Backlogs — data-calibrated per actual placed rates
    if backlogs == 1:
        adj_place *= 0.52          # actual rate: 23.4% → proportional penalty
        reasons.append("1 backlog present — moderate penalty applied")
    elif backlogs == 2:
        adj_place = min(adj_place, 0.08)   # actual rate: 6.0%
        reasons.append("2 backlogs — significant placement risk")
    elif backlogs >= 3:
        adj_place = min(adj_place, 0.08)   # actual rate: 5.5%
        reasons.append(f"{backlogs} backlogs — very high risk of rejection")

    # Employability penalty for very low CGPA
    if cgpa < 6.0:
        adj_emp *= 0.60
        reasons.append("CGPA < 6.0 reduces overall employability")

    return round(adj_place, 4), round(adj_emp, 4), reasons

def generate_feedback(profile: dict) -> dict:
    strengths, issues, improvements = [], [], []
    cgpa       = profile.get("cgpa", 0)
    backlogs   = profile.get("backlogs", 0)
    internships = profile.get("internships_done", 0)
    projects   = profile.get("projects_completed", 0)
    comm       = profile.get("communication_score", 0)
    apt        = profile.get("aptitude_score", 0)
    certs      = profile.get("num_certifications", 0)

    if cgpa >= 8.5:   strengths.append("Excellent CGPA (≥8.5) — opens doors to Tier-1 companies.")
    elif cgpa >= 7.0: strengths.append("Good CGPA (7.0–8.5) — eligible for most Tier-2 companies.")
    elif cgpa >= 6.0: improvements.append("Average CGPA (6.0–7.0) — compensate with strong projects & internships.")
    else:             issues.append("Low CGPA (<6.0) — many companies apply a hard filter here.")

    if backlogs == 0: strengths.append("Clean academic record — no active backlogs.")
    elif backlogs == 1: improvements.append("1 active backlog — clear it ASAP; some companies have a zero-backlog policy.")
    else: issues.append(f"{backlogs} active backlogs — hard filter for most companies. Resolve immediately.")

    if internships >= 2:   strengths.append(f"{internships} internships — strong industry exposure, highly valued.")
    elif internships == 1: improvements.append("1 internship — good start; a second one significantly boosts odds.")
    else:                  issues.append("No internship experience — pursue at least 1 before placements.")

    if projects >= 5:   strengths.append(f"{projects} projects — impressive portfolio for product companies.")
    elif projects >= 3: strengths.append(f"{projects} projects — solid project base.")
    elif projects >= 1: improvements.append(f"{projects} project(s) — aim for 3–5 end-to-end projects.")
    else:               issues.append("No projects — a portfolio is non-negotiable.")

    if comm >= 8:   strengths.append("Strong communication skills — important for client-facing roles.")
    elif comm >= 6: improvements.append("Average communication — work on presentation & articulation skills.")
    else:           issues.append("Weak communication — enrol in a spoken English or public speaking course.")

    if apt >= 80:   strengths.append("Strong aptitude (≥80) — clears most online assessment rounds.")
    elif apt >= 60: improvements.append("Average aptitude — practise quantitative & logical reasoning daily.")
    else:           issues.append("Low aptitude (<60) — practise on HackerRank / LeetCode urgently.")

    if certs >= 2:   strengths.append(f"{certs} certifications — validates domain knowledge.")
    elif certs == 1: improvements.append("1 certification — add 1–2 more relevant to your target domain.")
    else:            improvements.append("No certifications — free Coursera / NPTEL courses add credibility fast.")

    if profile.get("open_source_contributions", 0) > 0:
        strengths.append("Open-source contributions — highly valued by product companies.")
    if profile.get("research_papers", 0) > 0:
        strengths.append("Research paper(s) published — strong signal for R&D and PSU roles.")
    if profile.get("hackathons_participated", 0) >= 3:
        strengths.append(f"{profile['hackathons_participated']} hackathons — demonstrates initiative and team problem-solving.")

    return {"strengths": strengths, "critical_issues": issues, "improvements": improvements}

def get_tier_recommendation(emp_prob: float) -> str:
    if emp_prob >= 0.80: return "Tier-1 Product"
    if emp_prob >= 0.65: return "Tier-2 Product"
    if emp_prob >= 0.45: return "Tier-2 IT Services"
    if emp_prob >= 0.30: return "PSU/Research"
    return "Focus on skill-building before applying"

def score_student_for_company(student: dict, company: dict, company_name: str) -> dict:
    """
    Compute a fit score (0-100) for a student vs a company purely from
    rule-based heuristics (no BERT call — used for bulk shortlisting).
    """
    cgpa       = student["cgpa"]
    backlogs   = student["backlogs"]
    internships = student["internships"]
    projects   = student["projects"]
    comm       = student["comm"]
    tech_skills = student["tech_skills"]

    score = 100.0
    reasons = []

    # Hard CGPA filter
    if cgpa < company["min_cgpa"]:
        score *= 0.08
        reasons.append(f"CGPA {cgpa:.1f} below required {company['min_cgpa']:.1f}")
    else:
        surplus = min((cgpa - company["min_cgpa"]) / 2.0, 1.0)
        score  += surplus * 10

    # Backlogs
    if backlogs >= 2: score *= 0.50
    elif backlogs == 1:
        tier_enc = TIER_MAP.get(company["tier"], 1)
        if tier_enc >= 3: score *= 0.60

    # Internship
    if company["min_intern"] > 0 and internships < company["min_intern"]:
        score *= 0.70

    # Projects
    if projects >= company["min_proj"] + 2: score += 8
    elif projects >= company["min_proj"]:    score += 3
    else:                                    score *= 0.85

    # Communication
    if comm >= company["min_comm"]:
        score += min((comm - company["min_comm"]) * 1.5, 8)
    else:
        score *= 0.90

    # Skill match
    skill_match = compute_skill_match(tech_skills, company.get("skills", []))
    score += skill_match * 20

    score = max(0, min(100, score))

    # Label
    if score >= 75:   fit = "Strong Fit";  fit_color = "#2e7d32"
    elif score >= 55: fit = "Good Fit";    fit_color = "#1565c0"
    elif score >= 35: fit = "Possible Fit";fit_color = "#f57f17"
    else:             fit = "Reach";       fit_color = "#c0392b"

    return {
        "name":         company_name.title(),
        "score":        round(score, 1),
        "fit":          fit,
        "fit_color":    fit_color,
        "tier":         company["tier"],
        "domain":       company["domain"],
        "salary_lpa":   company.get("salary_lpa", "N/A"),
        "work_mode":    company.get("work_mode", "N/A"),
        "location":     company.get("location", "N/A"),
        "skill_match":  round(skill_match * 100, 1),
        "required_skills": company.get("skills", []),
        "reasons":      reasons,
    }

# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/status")
def status(): return jsonify({"loaded": models_loaded, "error": load_error, "device": str(DEVICE)})

@app.route("/api/company_info")
def company_info():
    name = request.args.get("name", "").lower().strip()
    info = KNOWN_COMPANIES.get(name)
    if info: return jsonify({"found": True, **info})
    return jsonify({"found": False})

@app.route("/api/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": f"Models not loaded. {load_error or ''}"}), 503

    data = request.get_json()
    mode = data.get("mode", "company")

    branch              = data.get("branch", "Computer Science")
    year                = int(data.get("year", 2024))
    cgpa                = float(data.get("cgpa", 7.0))
    tech_skills         = [s.strip() for s in data.get("tech_skills", []) if s.strip()]
    soft_skills         = [s.strip() for s in data.get("soft_skills", []) if s.strip()]
    certifications      = [s.strip() for s in data.get("certifications", []) if s.strip()]
    internships         = int(data.get("internships", 0))
    projects            = int(data.get("projects", 0))
    backlogs            = int(data.get("backlogs", 0))
    hackathons          = int(data.get("hackathons", 0))
    communication_score = int(data.get("communication_score", 5))
    aptitude_score      = float(data.get("aptitude_score", 60))
    soft_skills_rating  = int(data.get("soft_skills_rating", 5))
    tenth_percentage    = float(data.get("tenth_percentage", 70))
    twelfth_percentage  = float(data.get("twelfth_percentage", 70))
    open_source         = int(data.get("open_source", 0))
    research_papers     = int(data.get("research_papers", 0))

    num_tech = len(tech_skills)
    num_soft = len(soft_skills)
    num_cert = len(certifications)

    # ── Employability (GBT) — RAW unscaled data, matching training ──────────
    student_vec = np.array([[
        cgpa, tenth_percentage, twelfth_percentage,
        num_tech, num_soft, num_cert, internships, projects, backlogs,
        communication_score, aptitude_score, hackathons,
        open_source, research_papers, soft_skills_rating,
    ]], dtype=np.float32)
    # ✅ NO scaler — emp_model was trained on raw values (see notebook Cell 9)
    raw_emp_prob = float(emp_model.predict_proba(student_vec)[0, 1])

    # Default guardrail values for general mode
    company_min_cgpa = 6.0

    response = {
        "mode": mode,
    }

    if mode == "company":
        applied_company         = data.get("applied_company", "")
        company_tier_input      = data.get("company_tier", "Tier-2 IT Services")
        company_domain          = data.get("company_domain", "IT Services")
        company_required_skills = [s.strip() for s in data.get("company_required_skills", []) if s.strip()]
        company_min_cgpa        = float(data.get("company_min_cgpa", 6.0))
        company_min_intern      = int(data.get("company_min_intern", 0))
        company_min_proj        = int(data.get("company_min_proj", 1))
        company_min_comm        = int(data.get("company_min_comm", 5))

        known = resolve_company(applied_company)
        if known and not company_required_skills:
            company_required_skills = known["skills"]
            company_min_cgpa        = known["min_cgpa"]
            company_min_intern      = known["min_intern"]
            company_min_proj        = known["min_proj"]
            company_min_comm        = known["min_comm"]
            company_tier_input      = known["tier"]
            company_domain          = known["domain"]
            response["company_auto_filled"] = True
        elif not known:
            response["company_unknown"] = True

        skill_match = compute_skill_match(tech_skills, company_required_skills)
        tier_enc    = TIER_MAP.get(company_tier_input, 1)

        hybrid_raw = np.array([[
            cgpa, tenth_percentage, twelfth_percentage,
            num_tech, num_soft, num_cert, internships, projects, backlogs,
            communication_score, aptitude_score, hackathons,
            open_source, research_papers, soft_skills_rating,
            skill_match, company_min_cgpa, company_min_intern,
            company_min_proj, company_min_comm, tier_enc,
        ]])
        hybrid_scaled = scaler_hybrid.transform(hybrid_raw)

        profile_dict = {
            "branch": branch, "year": year, "cgpa": cgpa,
            "tech_skills": tech_skills, "soft_skills": soft_skills,
            "certifications": certifications, "internships": internships,
            "projects": projects, "backlogs": backlogs, "hackathons": hackathons,
            "communication_score": communication_score, "aptitude_score": aptitude_score,
            "applied_company": applied_company or "this company",
            "company_domain": company_domain, "company_tier": company_tier_input,
            "company_required_skills": company_required_skills, "skill_match_ratio": skill_match,
        }
        profile_text = build_profile_text(profile_dict)

        enc = tokenizer(profile_text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        with torch.no_grad():
            ids   = enc["input_ids"].to(DEVICE)
            mask  = enc["attention_mask"].to(DEVICE)
            tids  = enc["token_type_ids"].to(DEVICE)
            tab   = torch.tensor(hybrid_scaled, dtype=torch.float32).to(DEVICE)
            logit = placement_model(ids, mask, tids, tab)
            raw_placement_prob = float(torch.sigmoid(logit / TEMPERATURE).cpu().item())

        # ── Data-calibrated guardrails (from notebook) ──────────────────────
        placement_prob, emp_prob, guardrail_reasons = apply_guardrails(
            raw_placement_prob, raw_emp_prob, cgpa, backlogs, company_min_cgpa
        )
        placed = placement_prob >= THRESHOLD

        emp_cat, emp_color = get_emp_category(emp_prob)
        response.update({
            "employability":          round(emp_prob * 100, 1),
            "emp_category":           emp_cat,
            "emp_color":              emp_color,
            "tier_recommendation":    get_tier_recommendation(emp_prob),
            "placement_probability":  round(placement_prob * 100, 1),
            "placed":                 bool(placed),
            "skill_match":            round(skill_match * 100, 1),
            "hard_fail_reasons":      guardrail_reasons,
            "company_tier":           company_tier_input,
            "company_required_skills": company_required_skills,
            "feedback": generate_feedback({
                "cgpa": cgpa, "backlogs": backlogs,
                "communication_score": communication_score,
                "aptitude_score": aptitude_score,
                "num_technical_skills": num_tech,
                "internships_done": internships, "projects_completed": projects,
                "num_certifications": num_cert, "hackathons_participated": hackathons,
                "open_source_contributions": open_source, "research_papers": research_papers,
            }),
        })
    else:
        # General mode — no guardrail needed for skill match (no company)
        _, emp_prob_adj, reasons = apply_guardrails(0, raw_emp_prob, cgpa, backlogs, 6.0)
        emp_cat, emp_color = get_emp_category(emp_prob_adj)
        response.update({
            "employability":       round(emp_prob_adj * 100, 1),
            "emp_category":        emp_cat,
            "emp_color":           emp_color,
            "tier_recommendation": get_tier_recommendation(emp_prob_adj),
            "feedback": generate_feedback({
                "cgpa": cgpa, "backlogs": backlogs,
                "communication_score": communication_score,
                "aptitude_score": aptitude_score,
                "num_technical_skills": num_tech,
                "internships_done": internships, "projects_completed": projects,
                "num_certifications": num_cert, "hackathons_participated": hackathons,
                "open_source_contributions": open_source, "research_papers": research_papers,
            }),
        })

    return jsonify(response)


@app.route("/api/shortlist", methods=["POST"])
def shortlist():
    """
    Return top company matches for the student (no BERT — rule-based scoring).
    Supports optional filters: work_mode, min_salary_lpa, domain.
    """
    data = request.get_json()
    student = {
        "cgpa":       float(data.get("cgpa", 7.0)),
        "backlogs":   int(data.get("backlogs", 0)),
        "internships": int(data.get("internships", 0)),
        "projects":   int(data.get("projects", 0)),
        "comm":       int(data.get("communication_score", 5)),
        "aptitude":   float(data.get("aptitude_score", 60)),
        "tech_skills": [s.strip() for s in data.get("tech_skills", []) if s.strip()],
    }

    filter_mode    = data.get("work_mode_filter", "Any")        # Any / WFO / Hybrid / Remote
    filter_domain  = data.get("domain_filter", "Any").lower()   # Any / it services / product / fintech …
    top_n          = int(data.get("top_n", 8))

    results = []
    for name, company in KNOWN_COMPANIES.items():
        # Domain filter
        if filter_domain != "any" and filter_domain not in company["domain"].lower():
            continue
        # Work mode filter
        if filter_mode != "Any" and filter_mode.lower() not in company.get("work_mode", "").lower():
            continue

        result = score_student_for_company(student, company, name)
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    top     = results[:top_n]
    safe    = [r for r in results if r["fit"] in ("Strong Fit", "Good Fit")][:3]
    dream   = [r for r in results if r["tier"] in ("Tier-1 Product", "Tier-2 Product")][:3]

    return jsonify({"companies": top, "safe_bets": safe, "dream_picks": dream})


@app.route("/api/learning_path", methods=["POST"])
def learning_path():
    """Generate a personalised 30/60/90-day learning path."""
    data           = request.get_json()
    cgpa           = float(data.get("cgpa", 7.0))
    aptitude       = float(data.get("aptitude_score", 60))
    comm           = int(data.get("communication_score", 5))
    projects       = int(data.get("projects", 0))
    internships    = int(data.get("internships", 0))
    backlogs       = int(data.get("backlogs", 0))
    certs          = int(data.get("certifications", 0))
    tech_skills    = [s.lower().strip() for s in data.get("tech_skills", [])]
    target_tier    = data.get("target_tier", "Tier-2 IT Services")
    open_source    = int(data.get("open_source", 0))

    days30, days60, days90 = [], [], []
    certifications_suggested = []

    # ── Aptitude track ──────────────────────────────────────────────────────
    if aptitude < 60:
        days30.append({"icon": "🧮", "task": "Daily aptitude practice", "detail": "30 min/day on IndiaBIX — Number Series, Percentages, Profit & Loss", "priority": "Critical"})
        days60.append({"icon": "💻", "task": "HackerRank Problem Solving", "detail": "Complete 'Problem Solving (Basic)' certification — 2 problems/day", "priority": "High"})
    elif aptitude < 80:
        days30.append({"icon": "🧮", "task": "Aptitude polish", "detail": "Focus on Time & Work, Permutations — 20 min/day on PrepInsta", "priority": "Medium"})
        days60.append({"icon": "💻", "task": "LeetCode Easy/Medium streak", "detail": "15 problems/week — focus on Arrays, Strings, HashMap", "priority": "High"})

    # ── Communication ───────────────────────────────────────────────────────
    if comm < 6:
        days30.append({"icon": "🗣️", "task": "Communication bootcamp", "detail": "Daily 15-min spoken English exercises on ELSA Speak or Toastmasters recordings", "priority": "Critical"})
        days60.append({"icon": "🎤", "task": "Mock HR interviews", "detail": "Record yourself answering 'Tell me about yourself' and 'Why this company?' — refine weekly", "priority": "High"})

    # ── Projects ────────────────────────────────────────────────────────────
    if projects < 3:
        days30.append({"icon": "🛠️", "task": "Start a portfolio project", "detail": "Build a CRUD app with a REST API and deploy it on Vercel/Railway", "priority": "High"})
        days60.append({"icon": "🛠️", "task": "Complete 2nd project with ML/API", "detail": "Add a model inference endpoint or integrate a third-party API", "priority": "High"})
        days90.append({"icon": "🛠️", "task": "Open-source contribution or capstone", "detail": "Raise a PR on a GitHub repo with 100+ stars — adds credibility", "priority": "Medium"})

    # ── DSA Track ──────────────────────────────────────────────────────────
    has_dsa = any(k in tech_skills for k in ["data structures", "algorithms", "dsa", "leetcode"])
    if not has_dsa:
        days30.append({"icon": "📚", "task": "DSA Foundations", "detail": "Arrays, Linked Lists, Stacks, Queues — Striver's A2Z Sheet (first 50 problems)", "priority": "High"})
        days60.append({"icon": "📚", "task": "Trees, Graphs, DP", "detail": "Striver's Sheet sections 6–10; aim for 5 problems every day", "priority": "High"})
        days90.append({"icon": "📚", "task": "Competitive practice", "detail": "Join Codeforces Div-3 / LeetCode Weekly — consistent participation", "priority": "Medium"})

    # ── Certifications ──────────────────────────────────────────────────────
    if target_tier in ("Tier-1 Product", "Tier-2 Product"):
        certifications_suggested = [
            {"name": "AWS Cloud Practitioner", "platform": "AWS", "duration": "2–3 weeks", "link": "https://aws.amazon.com/certification/certified-cloud-practitioner/"},
            {"name": "Google Data Analytics", "platform": "Coursera", "duration": "6 months", "link": "https://www.coursera.org/professional-certificates/google-data-analytics"},
            {"name": "Meta Back-End Developer", "platform": "Coursera", "duration": "8 months", "link": "https://www.coursera.org/professional-certificates/meta-back-end-developer"},
        ]
        days60.append({"icon": "🎓", "task": "Start AWS Cloud Practitioner", "detail": "Free study path on AWS Skill Builder — exam voucher costs ~$100", "priority": "High"})
    elif target_tier == "Tier-2 IT Services":
        certifications_suggested = [
            {"name": "TCS iON Career Edge", "platform": "TCS iON", "duration": "2 weeks", "link": "https://learning.tcsionhub.in/hub/campus/"},
            {"name": "Infosys Springboard Python", "platform": "Infosys", "duration": "3 weeks", "link": "https://infyspringboard.onwingspan.com/"},
            {"name": "NPTEL Cloud Computing", "platform": "NPTEL", "duration": "12 weeks", "link": "https://nptel.ac.in/"},
        ]
        days30.append({"icon": "🎓", "task": "Complete TCS iON Career Edge", "detail": "Free course — directly recognised in TCS NQT assessments", "priority": "High"})
    elif target_tier == "PSU/Research":
        certifications_suggested = [
            {"name": "GATE Preparation", "platform": "Self-study", "duration": "6–12 months", "link": "https://gate.iitk.ac.in/"},
            {"name": "NPTEL VLSI / Embedded Systems", "platform": "NPTEL", "duration": "12 weeks", "link": "https://nptel.ac.in/"},
        ]
        days30.append({"icon": "📖", "task": "Begin GATE preparation", "detail": "Engineering Mathematics + your core branch — minimum 3 hr/day", "priority": "Critical"})

    # ── Open source ─────────────────────────────────────────────────────────
    if open_source == 0:
        days90.append({"icon": "🌐", "task": "First open-source PR", "detail": "Pick a 'good first issue' on GitHub — label filter: good-first-issue", "priority": "Medium"})

    # ── Internship / networking ─────────────────────────────────────────────
    if internships == 0:
        days60.append({"icon": "💼", "task": "Apply for virtual internships", "detail": "LinkedIn, Internshala, AngelList — apply to 5 listings/week minimum", "priority": "High"})

    # ── Backlogs ────────────────────────────────────────────────────────────
    if backlogs > 0:
        days30.insert(0, {"icon": "🚨", "task": "Clear active backlogs first", "detail": f"You have {backlogs} backlog(s) — prioritise these above everything else", "priority": "Critical"})

    # Default filler if nothing to suggest
    if not days90:
        days90.append({"icon": "🏆", "task": "Mock placement interview cycle", "detail": "Full 3-round mock: OA → Technical → HR — schedule with peers or use Pramp.com", "priority": "Medium"})

    return jsonify({
        "target_tier": target_tier,
        "days_30": days30,
        "days_60": days60,
        "days_90": days90,
        "certifications": certifications_suggested,
    })


@app.route("/api/resume_score", methods=["POST"])
def resume_score():
    """
    Score the student's profile as a recruiter would scan a résumé.
    Returns a score out of 100 with section-level breakdown.
    """
    data           = request.get_json()
    cgpa           = float(data.get("cgpa", 7.0))
    tenth          = float(data.get("tenth_percentage", 70))
    twelfth        = float(data.get("twelfth_percentage", 70))
    backlogs       = int(data.get("backlogs", 0))
    tech_skills    = [s.strip() for s in data.get("tech_skills", []) if s.strip()]
    soft_skills    = [s.strip() for s in data.get("soft_skills", []) if s.strip()]
    certifications = [s.strip() for s in data.get("certifications", []) if s.strip()]
    internships    = int(data.get("internships", 0))
    projects       = int(data.get("projects", 0))
    hackathons     = int(data.get("hackathons", 0))
    open_source    = int(data.get("open_source", 0))
    research       = int(data.get("research_papers", 0))
    comm           = int(data.get("communication_score", 5))

    sections = {}

    # Academics (25 pts)
    acad = 0
    if cgpa >= 9.0:    acad += 25
    elif cgpa >= 8.0:  acad += 20
    elif cgpa >= 7.0:  acad += 15
    elif cgpa >= 6.0:  acad += 8
    else:              acad += 2
    if backlogs == 0:  acad = min(acad + 3, 25)
    else:              acad = max(acad - backlogs * 4, 0)
    if tenth >= 85 or twelfth >= 85: acad = min(acad + 2, 25)
    sections["Academics"] = {"score": acad, "max": 25, "icon": "🎓"}

    # Skills (25 pts)
    skill_score = min(len(tech_skills) * 3, 18) + min(len(soft_skills) * 2, 7)
    sections["Skills"] = {"score": min(skill_score, 25), "max": 25, "icon": "💡"}

    # Experience (25 pts)
    exp = min(internships * 8, 16) + min(projects * 3, 9)
    sections["Experience"] = {"score": min(exp, 25), "max": 25, "icon": "💼"}

    # Extras (25 pts)
    extras = (
        min(len(certifications) * 4, 10) +
        min(hackathons * 2, 6) +
        min(open_source * 4, 6) +
        min(research * 4, 3)
    )
    sections["Extras & Certifications"] = {"score": min(extras, 25), "max": 25, "icon": "⭐"}

    total = sum(s["score"] for s in sections.values())

    if total >= 85:   grade = "A+"; grade_color = "#1565c0"
    elif total >= 70: grade = "A";  grade_color = "#2e7d32"
    elif total >= 55: grade = "B+"; grade_color = "#f57f17"
    elif total >= 40: grade = "B";  grade_color = "#e65100"
    else:             grade = "C";  grade_color = "#b71c1c"

    tips = []
    if sections["Academics"]["score"] < 18:
        tips.append("Add your CGPA prominently; consider showing an upward semester trend if applicable.")
    if sections["Skills"]["score"] < 18:
        tips.append("List at least 8 technical skills with proficiency levels (e.g., Python — Advanced).")
    if sections["Experience"]["score"] < 18:
        tips.append("Quantify your project impact: 'Reduced API latency by 30%' beats 'Built a REST API'.")
    if sections["Extras & Certifications"]["score"] < 15:
        tips.append("Add 1–2 industry-recognised certifications (AWS, Google, Microsoft) to boost this section.")
    if comm < 7:
        tips.append("Use strong action verbs (Designed, Optimised, Deployed) and keep bullet points concise.")

    return jsonify({
        "total": total,
        "grade": grade,
        "grade_color": grade_color,
        "sections": sections,
        "tips": tips,
    })


@app.route("/api/interview_tips", methods=["POST"])
def interview_tips():
    """Return tier-specific interview preparation tips."""
    data        = request.get_json()
    tier        = data.get("tier", "Tier-2 IT Services")
    company     = data.get("company", "").lower().strip()
    cgpa        = float(data.get("cgpa", 7.0))
    tech_skills = [s.lower().strip() for s in data.get("tech_skills", [])]

    rounds = []
    resources = []
    tips = []

    if tier == "Tier-1 Product":
        rounds = [
            {"name": "Online Assessment (OA)", "duration": "90 min", "focus": "2–3 LeetCode Medium/Hard DSA problems — timed", "weight": "Eliminatory"},
            {"name": "Technical Round 1", "duration": "60 min", "focus": "DSA deep-dive — Trees, Graphs, DP, System Design basics", "weight": "Critical"},
            {"name": "Technical Round 2 / System Design", "duration": "45–60 min", "focus": "Design URL Shortener / Ride-sharing — scalability focus", "weight": "Critical"},
            {"name": "Hiring Manager Round", "duration": "30–45 min", "focus": "Project depth, problem-solving approach, culture fit", "weight": "High"},
            {"name": "HR Round", "duration": "30 min", "focus": "Behavioural — STAR method, compensation, notice period", "weight": "Standard"},
        ]
        resources = [
            {"name": "Striver's A2Z DSA Sheet", "url": "https://takeuforward.org/strivers-a2z-dsa-course/"},
            {"name": "Grokking System Design", "url": "https://www.designgurus.io/course/grokking-the-system-design-interview"},
            {"name": "NeetCode 150", "url": "https://neetcode.io/practice"},
        ]
        tips = [
            "Think out loud — interviewers value your reasoning process over a perfect solution.",
            "For system design: always clarify requirements → estimate scale → identify bottlenecks → propose components.",
            f"At your CGPA of {cgpa:.1f}, lead with projects and internships — not just grades.",
            "Prepare 2–3 STAR stories for 'Tell me about a challenge' — use metrics whenever possible.",
        ]
    elif tier == "Tier-2 Product":
        rounds = [
            {"name": "Online Assessment (OA)", "duration": "60–75 min", "focus": "LeetCode Easy/Medium — 2 problems + MCQs on core CS", "weight": "Eliminatory"},
            {"name": "Technical Round 1", "duration": "45–60 min", "focus": "DSA (Arrays, DP, Graphs), one system design question", "weight": "Critical"},
            {"name": "Technical Round 2", "duration": "30–45 min", "focus": "Project deep-dive + OOP / DBMS concepts", "weight": "High"},
            {"name": "HR Round", "duration": "20–30 min", "focus": "Culture fit, role expectations, compensation", "weight": "Standard"},
        ]
        resources = [
            {"name": "LeetCode Top Interview 150", "url": "https://leetcode.com/studyplan/top-interview-150/"},
            {"name": "GeeksForGeeks Must-Do DSA", "url": "https://www.geeksforgeeks.org/must-do-coding-questions-for-companies/"},
            {"name": "Scaler Topics — System Design", "url": "https://www.scaler.com/topics/system-design/"},
        ]
        tips = [
            "Practise explaining your projects in 2 minutes — what problem, what tech, what outcome.",
            "Know your tech stack deeply — expect questions like 'Why did you choose React over Vue?'",
            "Review DBMS: normalisation, indexing, transactions, SQL joins.",
            "Prepare for OOP fundamentals: SOLID principles, design patterns (Singleton, Factory).",
        ]
    elif tier == "Tier-2 IT Services":
        rounds = [
            {"name": "Online Assessment (OA / NQT)", "duration": "90–120 min", "focus": "Aptitude, Reasoning, Verbal, Basic Coding (Easy problems)", "weight": "Eliminatory"},
            {"name": "Group Discussion (GD)", "duration": "15–20 min", "focus": "Current affairs, tech topics — demonstrate leadership & clarity", "weight": "High"},
            {"name": "Technical Interview", "duration": "30–45 min", "focus": "Core CS basics, 1–2 easy coding problems, project discussion", "weight": "High"},
            {"name": "HR Interview", "duration": "20–30 min", "focus": "Relocation, night shifts, STAR responses, salary expectations", "weight": "Standard"},
        ]
        resources = [
            {"name": "IndiaBIX — Aptitude", "url": "https://www.indiabix.com/"},
            {"name": "TCS NQT Mock Papers", "url": "https://www.tcsion.com/hub/campus/"},
            {"name": "PrepInsta — Company Specific", "url": "https://prepinsta.com/"},
        ]
        tips = [
            "Aptitude is the real filter — practise daily for 3–4 weeks before the OA.",
            "In GD, speak first or second with a structured point: 'I believe… because… for example…'",
            "Prepare answers for: strengths/weaknesses, why this company, where do you see yourself in 5 years.",
            f"With CGPA {cgpa:.1f}, you clear most IT services screens — focus energy on the aptitude round.",
        ]
    elif tier == "PSU/Research":
        rounds = [
            {"name": "GATE / Written Exam", "duration": "3 hours", "focus": "Core branch subjects + Engineering Mathematics", "weight": "Eliminatory"},
            {"name": "Document Verification", "duration": "1 day", "focus": "Certificates, backlogs, category — no academic gaps", "weight": "Eliminatory"},
            {"name": "Technical Interview", "duration": "30–45 min", "focus": "Core subjects, final year project, current affairs in your domain", "weight": "Critical"},
            {"name": "HR / Medical", "duration": "1 day", "focus": "Background check, medical fitness, joining formalities", "weight": "Standard"},
        ]
        resources = [
            {"name": "GATE Overflow", "url": "https://gateoverflow.in/"},
            {"name": "Made Easy Handwritten Notes", "url": "https://madeeasy.in/"},
            {"name": "NPTEL Core Branch Courses", "url": "https://nptel.ac.in/"},
        ]
        tips = [
            "GATE score is the primary filter — previous year papers are the best preparation.",
            "Know your final year project inside out — PSU interviewers dig deep into it.",
            "Backlogs can be a disqualifier — check each PSU's eligibility criteria carefully.",
            "Current affairs in your domain (e.g., defence technology, space missions) often come up.",
        ]

    company_specific = []
    known = KNOWN_COMPANIES.get(company, {})
    if known:
        required = known.get("skills", [])
        student_has   = [s for s in required if any(s.lower() in t for t in tech_skills)]
        student_lacks = [s for s in required if s not in student_has]
        if student_lacks:
            company_specific.append(f"You're missing: {', '.join(student_lacks[:4])} — these are expected at {company.title()}.")
        if student_has:
            company_specific.append(f"You already have: {', '.join(student_has[:4])} — mention these prominently in the interview.")

    return jsonify({
        "tier": tier,
        "rounds": rounds,
        "resources": resources,
        "tips": tips,
        "company_specific": company_specific,
    })


if __name__ == "__main__":
    load_models()
    app.run(debug=False, host="0.0.0.0", port=5000)
