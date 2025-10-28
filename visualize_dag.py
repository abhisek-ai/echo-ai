print("""
EchoAI Pipeline DAG Structure:
==============================

    [acquire_data]
           ↓
    [preprocess_data]
           ↓
    [feature_engineering]
           ↓
      ┌────┼────┐
      ↓    ↓    ↓
[validate] [bias] [anomalies]
      ↓    ↓    ↓
      └────┼────┘
           ↓
    [generate_report]

DAG Properties:
- Schedule: Daily (@daily)
- Retries: 2
- Retry Delay: 5 minutes
- Email on Failure: Yes
""")
