# 🚀 Multi-Agent Data Analysis System (MAS)
### *Data Agents Track – Kaggle Agents Intensive Capstone*

---

# ## ✨ Title  
**Multi-Agent Data Analysis System (MAS): An Automated, Transparent, and Scalable Workflow for Data Science**

---

# ## 📌 Subtitle  
An AI-driven agent team that performs end-to-end EDA, AutoML, evaluation, and logging—powered by goal-driven autonomy.

---

# # 1. Problem Statement  

Modern data analysis is a multi-step, expertise-heavy process involving:

- Exploratory Data Analysis (EDA)  
- Visualization  
- Data cleaning  
- Creating train/test splits  
- Selecting algorithms  
- Hyperparameter tuning  
- Model evaluation  

For non-technical researchers—biologists, clinicians, educators—these tasks are:

- Time-consuming  
- Prone to human error  
- Difficult to reproduce  
- Dependent on programming expertise  

Even trained analysts struggle with:

- Maintaining consistency  
- Re-running pipelines for new datasets  
- Eliminating manual errors  
- Keeping experiments reproducible  

There is a growing need for:

- **Automation**  
- **Explainability**  
- **Modular workflows**  
- **Reproducibility**  

The **Multi-Agent Data Analysis System (MAS)** provides a solution:  
Users can request an analysis pipeline using natural language such as:

"train model with automl and show eda"

The system autonomously performs:

- EDA  
- Visualization  
- Train-test splitting  
- AutoML  
- Evaluation  
- Logging  

making high-quality data analysis accessible to everyone.

---

# # 2. Why Agents?  

Traditional automation scripts fail because they are rigid and require manual updates.  
A **multi-agent architecture** works better because it provides:

---

## ✔ Modularization  
Each agent focuses on **one responsibility**, simplifying code and reasoning.

---

## ✔ Specialization  
Each agent possesses unique skills:

- **ResearchAgent** → statistics + visualization  
- **ModelingAgent** → model selection + tuning  
- **EvalAgent** → metrics + confusion matrix  

---

## ✔ Coordination  
The **CoordinatorAgent** interprets user goals and ensures each agent runs in the correct order.

---

## ✔ Explainability  
The **Memory** system logs all actions, making the workflow transparent and easy to audit.

---

## ✔ Scalability  
New agents—such as SHAP explainability, deployment, or data cleaning—can be added easily.

---

**In short:**

> Agents provide the most flexible, interpretable, and human-friendly solution to automated data analysis.

---

# # 3. What I Created — System Architecture  

The **Multi-Agent Data Analysis System (MAS)** contains six interacting agents.

---

## 🔍 1. ResearchAgent (AutoEDA)
**Role:** Exploratory Data Analyst  
**Responsibilities:**

- Generate summary statistics  
- Check missing values  
- Create histograms  
- Build correlation heatmaps  
- Provide EDA insights  

---

## 🤖 2. ModelingAgent (SimpleAutoML)
**Role:** Machine Learning Model Selector  
**Responsibilities:**

- Train multiple classifiers  
- Perform hyperparameter tuning  
- Select the best model  
- Return accuracy and estimator  

Models considered include:

- RandomForest  
- SVC  
- Logistic Regression  
- KNN  
- DecisionTree  

---

## 📈 3. EvalAgent  
**Role:** Evaluator  
**Responsibilities:**

- Accuracy score  
- Classification report  
- Confusion matrix plot  

---

## 🧭 4. CoordinatorAgent  
**Role:** Planner and Controller  
**Responsibilities:**

- Parse user instruction  
- Decide workflow  
- Call agents in correct order  
- Manage outputs between agents  

---

## 🗂 5. Memory  
**Role:** Execution Trace Logger  
**Responsibilities:**

- Record every agent action  
- Provide a transparent log  
- Assist debugging and explainability  

Example log:
ResearchAgent → Completed EDA.
ModelingAgent → Best model SVC (97%).
EvalAgent → Generated confusion matrix.

---

## 🔧 6. ToolRegistry  
**Role:** Shared Tool Provider  
**Responsibilities:**

- Dataset loader  
- Utility functions  
- Environment tools  

---

## 🏗 Architecture Diagram (Markdown)




---

# # 4. Demo — Running the MAS on the Iris Dataset  

### **User Input**

"train model with automl and show eda"

---

### **System Workflow**

1. **Perform EDA**  
   - Summary statistics  
   - Distribution plots  
   - Correlation heatmap  

2. **Create Train/Test Split**

3. **Run AutoML**  
   - Test multiple models  
   - Grid search tuning  

4. **Pick Best Model**

5. **Evaluate Model**  
   - Generate accuracy  
   - Produce classification report  
   - Create confusion matrix  

6. **Log all actions** using Memory  

---

# ## 📈 Demo Results (Iris Example)

- **Best Model:** SVC or RandomForest  
- **Accuracy:** *95% – 100%*  
- **Evaluation Outputs:**  
  - Confusion matrix image  
  - Accuracy and report  
  - Full agent log  

The workflow completes **autonomously** without writing any code.

---

# # 5. The Build — Tools, Technologies, and Implementation  

### **Languages & Libraries**
- Python 3.10+  
- scikit-learn  
- pandas  
- numpy  
- matplotlib / seaborn  
- Custom agent classes  
- Logging module  

---

### **Concepts Demonstrated (Matching Kaggle Requirements)**

| Kaggle Requirement | How MAS Satisfies It |
|--------------------|-----------------------|
| Multi-agent system | Six specialized agents |
| Coordination | CoordinatorAgent orchestrates workflow |
| Data interaction | EDA, AutoML, evaluation |
| Explainability | Memory logs every action |
| Architecture clarity | Fully modular design |
| Innovation | Natural-language-driven ML pipeline |

---

### **Code Design Principles Applied**

- **Single Responsibility Principle**  
- **Loosely coupled agent modules**  
- **Trace logging for transparency**  
- **Reusable utilities and data loaders**  
- **Goal-driven workflow execution**  

---

# # 6. Impact & Value  

### 🎯 **For Non-Technical Researchers**
- Zero coding required  
- Automated ML pipeline  
- Easy-to-understand outputs  

### 🎯 **For Data Scientists**
- Fast prototyping  
- Reproducible pipelines  
- Extensible agent ecosystem  

### 🎯 **For Educators**
- Teaches ML workflow  
- Demonstrates agent-based system design  

---

# # 7. Why This Project Matters  

This project empowers:

- Biologists  
- Environmental researchers  
- Healthcare professionals  
- Students  
- Analysts  

who need to run machine learning but may not have a technical background.

It bridges the gap between **domain expertise** and **technical execution**.

With automated agent workflows, users can:

- Run analyses faster  
- Avoid errors  
- Standardize entire ML pipelines  
- Focus on insights instead of code  

---

# # 8. If I Had More Time (Future Work)  

Planned improvements include:

---

### 🔹 **SHAP Explainability Agent**  
To generate feature importance explanations.

### 🔹 **Data Cleaning Agent**  
For outlier removal, encoding, handling missing data.

### 🔹 **Model Deployment Agent**  
To publish trained models as services.

### 🔹 **Dashboard Agent**  
Automatically create Streamlit dashboards.

### 🔹 **Support Regression & Clustering**  
Extend AutoML to more ML task types.

### 🔹 **More Advanced ML Models**  
Add XGBoost, CatBoost, LightGBM.

---

# # 9. Conclusion  

The **Multi-Agent Data Analysis System (MAS)** is a modular, transparent, and scalable AI-driven pipeline that:

- Interprets user goals  
- Performs EDA  
- Executes AutoML  
- Evaluates models  
- Logs all actions  

This system meets and exceeds the expectations of the **Data Agents Track** by demonstrating:

- Real-world practical automation  
- Explainability through logging  
- A clean multi-agent architecture  
- High usability for non-programmers  

MAS shows how multi-agent systems can transform data science into an accessible, reproducible, and autonomous process.

---
