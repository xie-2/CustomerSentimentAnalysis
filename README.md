# Customer Experience Analysis

This project analyzes customer experience data to predict customer retention and generate insights through various visualizations.

## Project Structure

```
EmailSentimentAnalysis/
├── src/
│   ├── fetchemails.py      # Email fetching functionality
│   ├── preprocess.py       # Data preprocessing
│   ├── trainmodel.py       # Model training
│   ├── predict.py          # Making predictions
│   ├── visualize.py        # Data visualization
│   └── utils.py           # Utility functions
├── data/                   # Data directory (not in git)
└── requirements.txt        # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your customer experience data in the `data/` directory as `customer_experience_data.csv`

## Usage

1. Train the model:
```bash
python3 src/trainmodel.py
```

2. Make predictions:
```bash
python3 src/predict.py --file data/customer_experience_data.csv
```

3. Generate visualizations:
```bash
python3 src/visualize.py
```

## Data Format

The input CSV file should contain the following columns:
- Customer_ID
- Age
- Gender
- Location
- Num_Interactions
- Feedback_Score
- Products_Purchased
- Products_Viewed
- Time_Spent_on_Site
- Satisfaction_Score
- Retention_Status
- Gender_Encoded
- Location_Encoded
- Retention_Status_Encoded

## Output

The analysis generates:
- Trained model files in `data/`
- Prediction results in `data/predictions.csv`
- Various visualizations in `data/`:
  - distributions.png
  - correlation_matrix.png
  - segment_analysis.png
  - demographic_analysis.png
  - interaction_analysis.png
  - time_analysis.png
  - product_engagement.png

## License
MIT License