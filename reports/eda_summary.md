# EDA summary

- Customers: **7,043**, features: **23**
- Overall churn rate: **26.5%** (class imbalance ~2.8:1)

## Churn rate by key segments

- **Contract**: highest `Month-to-month` (43%), lowest `Two year` (3%) -> 15.1x difference
- **tenure_group**: highest `0-6m` (53%), lowest `49m+` (10%) -> 5.6x difference
- **PaymentMethod**: highest `Electronic check` (45%), lowest `Credit card (automatic)` (15%) -> 3.0x difference
- **InternetService**: highest `Fiber optic` (42%), lowest `No` (7%) -> 5.7x difference
- **TechSupport**: highest `No` (42%), lowest `No internet service` (7%) -> 5.6x difference

## Medians: stayed vs churned

| Churn   |   tenure |   MonthlyCharges |   num_services |
|:--------|---------:|-----------------:|---------------:|
| Stayed  |       38 |           64.425 |              4 |
| Churned |       10 |           79.65  |              4 |

## Takeaways

1. Month-to-month contracts are the single biggest churn driver; long contracts barely churn.
2. Churn is front-loaded: the first 6-12 months are the danger zone.
3. Electronic-check payers and fiber-optic customers churn far more than average.
4. Customers without tech support / online security churn more -> service bundles retain.
5. Churners pay *higher* monthly bills on average -> price sensitivity matters.