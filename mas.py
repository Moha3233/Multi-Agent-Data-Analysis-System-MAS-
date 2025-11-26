"""
Multi-Agent Data Analysis System (MAS)
Author: Mohan G. Duratkar

This script implements a collaborative multi-agent system for automated:
- Exploratory Data Analysis (EDA)
- Train/Test Splitting
- Simple AutoML
- Model Evaluation
- Logging / Memory Tracking

Agents:
1. ResearchAgent     → Handles EDA.
2. ModelingAgent     → Runs AutoML (GridSearchCV).
3. EvalAgent         → Metrics & Confusion Matrix.
4. CoordinatorAgent  → Interprets user intent & orchestrates workflow.
5. Memory            → Stores logs.
6. ToolRegistry      → Dataset loading utilities.
"""

# ============================================================
# Imports
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import io
import contextlib


# ============================================================
# Memory Module - Stores All Logs
# ============================================================

class Memory:
    def __init__(self):
        self.logs = []

    def add(self, msg):
        print("LOG:", msg)
        self.logs.append(msg)

    def show(self):
        return "\n".join(self.logs)


# ============================================================
# Tool Registry - Utility Functions
# ============================================================

class ToolRegistry:
    def __init__(self):
        pass

    def load_dataset(self, name="iris"):
        """
        Load dataset by name. Defaults to Iris.
        """
        from sklearn.datasets import load_iris

        if name == "iris":
            data = load_iris(as_frame=True)
            df = data.frame
            df["target"] = data.target
            return df

        raise ValueError("Dataset not supported yet.")


# ============================================================
# ResearchAgent - EDA
# ============================================================

class ResearchAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    def run_eda(self, df):
        self.memory.add("ResearchAgent: Running EDA.")

        # Capture EDA output
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print("\n===== DATA HEAD =====")
            print(df.head())

            print("\n===== INFO =====")
            print(df.info())

            print("\n===== DESCRIBE =====")
            print(df.describe())

            print("\n===== MISSING VALUES =====")
            print(df.isnull().sum())

        eda_report = buffer.getvalue()

        # Basic visualization
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

        return eda_report


# ============================================================
# ModelingAgent - AutoML
# ============================================================

class ModelingAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    def automl(self, X_train, y_train):
        self.memory.add("ModelingAgent: Starting model selection.")

        models = {
            "RandomForest": (
                RandomForestClassifier(),
                {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
            ),
            "SVC": (
                SVC(),
                {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
            )
        }

        best_model = None
        best_score = -1
        best_name = None

        for name, (model, params) in models.items():
            self.memory.add(f"ModelingAgent: Running GridSearchCV for {name}")

            grid = GridSearchCV(model, params, cv=3)
            grid.fit(X_train, y_train)

            score = grid.best_score_

            if score > best_score:
                best_score = score
                best_model = grid.best_estimator_
                best_name = name

        self.memory.add(f"ModelingAgent: Best model is {best_name} with score {best_score:.3f}")

        return best_model, best_name, best_score


# ============================================================
# EvalAgent - Evaluation & Plots
# ============================================================

class EvalAgent:
    def __init__(self, memory: Memory):
        self.memory = memory

    def evaluate(self, model, X_test, y_test):
        self.memory.add("EvalAgent: Evaluating the model.")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return acc, report, cm


# ============================================================
# CoordinatorAgent - Orchestrates All Agents
# ============================================================

class CoordinatorAgent:
    def __init__(self):
        self.memory = Memory()
        self.tools = ToolRegistry()
        self.researcher = ResearchAgent(self.memory)
        self.modeler = ModelingAgent(self.memory)
        self.evaluator = EvalAgent(self.memory)

    def run(self, user_goal):
        """
        Interpret natural-language goal and run a full pipeline.
        """
        self.memory.add(f"CoordinatorAgent: Received user goal → {user_goal}")

        # Load dataset
        df = self.tools.load_dataset("iris")
        self.memory.add("CoordinatorAgent: Loaded Iris dataset.")

        # Run EDA
        eda_report = self.researcher.run_eda(df)

        # Prepare data
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # AutoML
        best_model, model_name, score = self.modeler.automl(X_train, y_train)

        # Evaluation
        acc, report, cm = self.evaluator.evaluate(best_model, X_test, y_test)

        # Final logs
        final_logs = self.memory.show()

        return {
            "EDA_Report": eda_report,
            "Best_Model": model_name,
            "CV_Score": score,
            "Accuracy": acc,
            "Classification_Report": report,
            "Confusion_Matrix": cm,
            "Logs": final_logs
        }


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    system = CoordinatorAgent()
    results = system.run("train model with automl and show eda")

    print("\n===== FINAL RESULTS =====")
    print("Best Model:", results["Best_Model"])
    print("Accuracy:", results["Accuracy"])
    print("\nClassification Report:\n", results["Classification_Report"])
    print("\nLogs:\n", results["Logs"])
