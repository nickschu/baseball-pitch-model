# MLB Pitch Prediction Model

This project implements a baseball pitch prediction model for Major League Baseball (MLB). Utilizing historical pitch data sourced from MLB's Statcast system, the model predicts the type of the next pitch thrown (e.g., 4-seam fastball, 2-seam fastball, changeup, curveball, slider).

## Model Overview

The prediction model is built upon a modified FT-Transformer architecture, adapted to handle sequential tabular data. This model leverages both categorical (e.g., pitcher, batter, pitch count, inning) and continuous data (e.g., pitch velocity, spin rate, release angle) from previous pitches to capture contextual and temporal dependencies for predictions.

## Data Source

- **MLB Statcast Data**: Detailed pitch-by-pitch metrics, including: velocity; spin rate; pitch type; batter, pitcher, and catcher identities; game situation; and other relevant factors.

## Setup

To use this model install the dependencies in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

