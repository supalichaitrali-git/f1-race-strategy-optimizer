# F1 Race Strategy Optimizer

An F1 race strategy optimization system that uses historical Formula 1 race data, statistical tire analysis, machine learning, and strategy simulation to estimate effective tire and pit-stop strategies.

## Project Goal

Given a driver's current race state, the system evaluates possible tire and pit-stop strategies and recommends the strategy with the lowest estimated remaining race time.

Example:

- Driver: VER
- Race: Italian GP
- Current Lap: 30
- Current Tire: Medium
- Position: P4

The system may compare strategies such as:

- Medium → Hard
- Medium → Medium → Soft
- Soft → Medium
- Hard → Medium

## Technology Stack

- Python
- FastF1
- Pandas
- NumPy
- SQLite
- Scikit-learn
- FastAPI
- Streamlit
- Plotly
- Pytest
- Docker
- GitHub Actions
- Terraform
- AWS (optional)

## Architecture

```text
F1 Data
   |
   v
FastF1
   |
   v
Data Pipeline
   |
   v
SQLite
   |
   +----------------+
   |                |
   v                v
Tire Analysis     ML Model
   |                |
   +-------+--------+
           |
           v
   Strategy Simulator
           |
           v
        FastAPI
           |
           v
      Streamlit
           |
           v
          User
