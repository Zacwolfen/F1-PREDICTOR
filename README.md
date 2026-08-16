# F1 Race Prediction Model

A machine learning project built around Formula 1 data. The goal is to use historical race, driver, team, lap, and weather information to build a pipeline that can eventually predict race outcomes.

The project uses **FastF1** to collect session data and turns it into structured datasets that can be used for feature engineering and machine learning experiments.

## Project Overview

The basic idea is pretty simple: look at what has happened in previous races and see how useful that information is when predicting future results.

The current pipeline looks like this:

1. Download historical race-session data using FastF1.
2. Store the raw race, lap, weather, and metadata files.
3. Build a feature dataset from the collected data.
4. Create rolling and historical performance features.
5. Prepare the training dataset.
6. Train and evaluate different machine learning models.
7. Keep improving the features and predictions as the project develops.

## Data

The pipeline currently works with Formula 1 race data from **2018 onward**, with the project configured to collect seasons through **2026**.

The generated dataset includes things like:

- Driver
- Team
- Starting grid position
- Finishing position
- Championship points
- Season and race round
- Circuit and location
- Average, fastest, and median lap times
- Lap consistency
- Completed laps
- Sector performance
- Tyre life and stint information
- Air temperature
- Track temperature
- Humidity
- Pressure
- Wind speed
- Rainfall
- Driver form over the previous 3, 5, and 10 races
- Driver points over the previous 5 races
- Recent grid performance
- Team form
- Positions gained
- Average recent position gain
- DNF history
- Track history

## Feature Engineering

Instead of just feeding raw race results into a model, the project creates features that describe recent and historical performance.

Some of the current features are:

- `DriverForm3`
- `DriverForm5`
- `DriverForm10`
- `DriverPoints5`
- `DriverGrid5`
- `TeamForm5`
- `PositionsGained`
- `AvgGain5`
- `DNFRate10`
- `TrackHistory`

The idea is to give the model some context about how a driver and team have been performing recently, along with how they have historically performed at a particular circuit.

## Project Structure

```text
F1-PREDICTOR/
│
├── data/
│   ├── raw/
│   └── logs/
│
├── cache/
│
├── notebooks/
│
├── src/
│   ├── features/
│   ├── models/
│   └── preprocessing/
│
├── models/
│
├── .venv/
│
└── README.md
```

The exact contents of individual folders may change as the project grows.

## Tech Stack

- **Python**
- **FastF1** — Formula 1 data collection
- **Pandas** — data processing
- **NumPy** — numerical operations
- **Jupyter Notebook** — experimentation and analysis
- **scikit-learn / ML libraries** — model development
- **Git & GitHub** — version control

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd F1-PREDICTOR
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

On macOS / Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Run the data collection pipeline

The project includes a FastF1-based data downloader that creates the raw race-session datasets.

```bash
python3 src/download_all_data.py
```

If the downloader is stored somewhere else in the repository, adjust the path accordingly.

## Current Dataset Pipeline

The feature-generation process currently produces a structured dataset with **479 rows and 28 initial features**.

After adding rolling and historical features, the training dataset contains **479 rows and 39 features**.

The overall workflow is:

**Raw Data → Feature Dataset → Training Dataset → ML Model → Prediction**

Keeping these stages separate makes it easier to experiment with different features, preprocessing methods, and models without having to rebuild the entire pipeline every time.

## Goals

The main goals of the project are:

- Build a reliable F1 race prediction pipeline.
- Engineer useful driver, team, circuit, and weather features.
- Compare different machine learning approaches.
- Improve prediction performance through feature engineering.
- Avoid data leakage by using only information that would have been available before the race being predicted.
- Eventually generate predictions for upcoming Grand Prix weekends.

## Future Improvements

Some things I want to add as the project develops:

- Add qualifying-session features.
- Improve weather and track-condition features.
- Add more circuit-specific performance metrics.
- Experiment with different regression and classification models.
- Add hyperparameter tuning.
- Implement proper time-based validation.
- Build automated prediction reports.
- Create dashboards for predictions and feature importance.
- Automate the pipeline for upcoming Grand Prix weekends.

## Disclaimer

This is an educational and experimental machine learning project.

Formula 1 results depend on a lot of things that are difficult to predict, including strategy, incidents, weather, reliability, safety cars, and team decisions. Because of that, model predictions should be treated as statistical estimates rather than guaranteed outcomes.

## About

**Kaushik P**

Computer Science Engineering student interested in:

- Machine Learning
- Data Science
- Formula 1 Analytics
- Software Development
- Game Development
