# Data

Raw data is **not** committed to this repo. Download it locally.

## Download

1. Create a Kaggle account and accept the competition rules:
   https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Download and unzip into this folder. You need at minimum:
   - `application_train.csv`
   - `application_test.csv`
   - (optional) `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`

Or via the Kaggle CLI:

\`\`\`bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data/
cd data && unzip home-credit-default-risk.zip
\`\`\`
