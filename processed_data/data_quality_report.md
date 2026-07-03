# Data Quality Report

## Cleaning Process Summary
- **Initial rows**: 525461
- **Duplicates removed**: 6865
- **Bad Debt records removed**: 3
- **Zero-price records removed**: 3681
- **Test/Bank Charges records removed**: 78
- **Final rows**: 514834

## Columns Dropped
- `store_id`, `category`, `discount`, `payment_method` (100% empty)

## Data Type Corrections
- `date` cast to datetime.
- `transaction_id`, `product_id`, `region` stripped of leading/trailing whitespace.
- `customer_id` cast to integer strings (e.g. '12345'), preserving `NaN` for missing values.

## Business Decisions
- Retained postage and carriage charges (`POST`, `DOT`, `C2`).
- Retained customer returns and cancellations (represented by negative quantities and transaction IDs starting with 'C').
