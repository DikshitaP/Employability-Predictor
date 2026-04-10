<h1 align="center">🎯 Student Placement Prediction System</h1>

<p align="center">
A Deep Learning + Machine Learning Hybrid System for predicting<br>
<b>Placement Probability</b> and <b>Student Employability</b>
</p>

<hr>

<h2>🚀 Overview</h2>

<p>
This project is an intelligent placement prediction system that solves <b>two key problems</b>:
</p>

<ul>
<li><b>Company-Specific Prediction:</b> Will a student get placed in a particular company?</li>
<li><b>General Employability:</b> How employable is the student overall?</li>
</ul>

<p>
A student may be highly employable but still fail in a specific company due to skill mismatch — this system captures that nuance using a <b>dual-model architecture</b>.
</p>

<hr>

<h2>🧠 Model Architecture</h2>

<h3>1️⃣ Placement Prediction Model (BERT + LSTM Hybrid)</h3>

<ul>
<li>BERT (<code>bert-base-uncased</code>) processes a natural language student profile</li>
<li>Last 4 layers of BERT are fine-tuned</li>
<li>Output passed through a <b>2-layer LSTM (hidden size = 256)</b></li>
<li>Final hidden state → 128-dim text embedding</li>
<li>21 tabular features → MLP → 64-dim embedding</li>
<li>Both embeddings concatenated → classifier</li>
</ul>

<p><b>Final Input Representation:</b> 128 (text) + 64 (tabular) = 192 dimensions</p>

<h4>⚙️ Training Details</h4>
<ul>
<li>Loss Function: <b>Focal Loss (γ = 2, α = 1.2)</b></li>
<li>Optimizer: AdamW with differential learning rates</li>
<li>BERT: 1e-5 | LSTM: 3e-4 | Others: 5e-4</li>
<li>Scheduler: Linear Warmup</li>
<li>Epochs: 6 with Early Stopping</li>
</ul>

<hr>

<h3>2️⃣ Employability Model (Gradient Boosting)</h3>

<ul>
<li>Model: <b>GradientBoostingClassifier</b></li>
<li>Trained ONLY on <b>Tier-2 IT Services</b> data</li>
<li>Reason: Provides a neutral baseline for student-only evaluation</li>
<li>Features: 15 student-controlled attributes</li>
</ul>

<hr>

<h2>📊 Dataset</h2>

<ul>
<li>Size: <b>50,000 records</b></li>
<li>Type: Synthetic but realistic</li>
<li>Includes:</li>
<ul>
<li>Academic data (CGPA, 10th, 12th)</li>
<li>Projects, internships, hackathons</li>
<li>Skills (technical + soft)</li>
<li>Company context (tier, requirements, domain)</li>
</ul>
<li>Target: <code>placed (0/1)</code></li>
</ul>

<hr>

<h2>⚙️ Feature Engineering</h2>

<ul>
<li><b>Hybrid Features (21)</b> → Student + Company context</li>
<li><b>Student Features (15)</b> → Only student attributes</li>
<li>Tier encoding: <br>
Tier-1 Product = 4 → Tier-2 IT Services = 1</li>
<li>Normalization: <b>StandardScaler (fit only on training set)</b></li>
</ul>

<hr>

<h2>🎯 Calibration & Thresholding</h2>

<ul>
<li>Temperature Scaling applied to logits</li>
<li>Optimized using <b>L-BFGS</b></li>
<li><b>Temperature (T): 0.7862</b></li>
<li>Final classification threshold: <b>0.53</b></li>
</ul>

<p>
This ensures probabilities are <b>well-calibrated and not overconfident</b>.
</p>

<hr>

<h2>🛡️ Post-Prediction Guardrails</h2>

<p>
Neural networks can sometimes ignore hard constraints. To fix this:
</p>

<ul>
<li>CGPA below company requirement → probability capped at <b>8%</b></li>
<li>Backlogs penalties applied based on real dataset statistics</li>
<li>Ensures realistic and trustworthy predictions</li>
</ul>

<p>
<b>This is a key innovation of the system.</b>
</p>

<hr>

<h2>🔍 Explainability (SHAP)</h2>

<ul>
<li>SHAP used for interpretability</li>
<li>Direct SHAP on GBT (employability model)</li>
<li>Proxy GBT model used for hybrid tabular explanation</li>
</ul>

<p>
Helps answer: <i>"Why did the model give this prediction?"</i>
</p>

<hr>

<h2>💡 Features of the System</h2>

<ul>
<li>📈 Placement Probability Prediction</li>
<li>🧠 Employability Score</li>
<li>🏢 Company Fit Analysis</li>
<li>📊 Resume Scoring System</li>
<li>🎯 Company Shortlisting</li>
<li>🧭 30-60-90 Day Learning Path</li>
<li>🎤 Interview Preparation Guide</li>
<li>📌 Personalized Feedback Engine</li>
</ul>

<hr>

<h2>🧪 Evaluation Metrics</h2>

<ul>
<li>Accuracy</li>
<li>Precision & Recall</li>
<li>F1 Score</li>
<li>AUC-ROC</li>
<li>Log Loss</li>
<li>Matthews Correlation Coefficient (MCC)</li>
</ul>

<p>
MCC is used for balanced evaluation even in imbalanced datasets.
</p>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
<li><b>Backend:</b> Flask</li>
<li><b>Deep Learning:</b> PyTorch, Transformers (BERT)</li>
<li><b>Machine Learning:</b> Scikit-learn</li>
<li><b>Frontend:</b> HTML, CSS, JavaScript</li>
<li><b>Deployment:</b> Hugging Face + GitHub</li>
</ul>

<hr>

<h2>📦 Project Structure</h2>

<pre>
project/
│── app.py
│── templates/
│── static/
│── placement_model_best.pt
│── scaler files (.pkl)
│── emp_model.pkl
</pre>

<hr>

<h2>⚡ How to Run</h2>

<pre>
pip install -r requirements.txt
python app.py
</pre>

<p>Then open: <b>http://localhost:5000</b></p>

<hr>

<h2>🌟 Key Highlights</h2>

<ul>
<li>Hybrid Deep Learning + ML approach</li>
<li>Realistic constraints via guardrails</li>
<li>Dual prediction system (specific + general)</li>
<li>Explainable AI integration (SHAP)</li>
<li>Industry-ready feature set</li>
</ul>

<hr>

<h2 align="center"> - Dikshita Pimpale</h2>