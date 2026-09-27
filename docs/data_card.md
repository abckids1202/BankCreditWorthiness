# Data card

## Primary source

UCI Default of Credit Card Clients: https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients

## Unit of observation

An existing credit-card customer record with six months of repayment, bill, and payment history.

## Target

`default.payment.next.month`, normalized internally to `default`, indicates whether the customer defaulted in the following month.

## Feature treatment

ID and demographic fields are not used by the prediction model. Engineered features summarize utilization, late payments, payment-to-bill ratios, balance trends, and account stability using only the available historical months.

## Known limitations

The dataset is historical, geographically limited, and not a new-loan application dataset. Its target and feature availability should not be assumed to match a bank’s underwriting process.

## Dataset governance

Record the source URL, license, SHA-256 hash, row count, target rate, and configuration in every generated model artifact. Never add real customer data to this repository.

## Alternate research datasets

The repository also supports separate experiments (never merged with the primary
dataset):

| Adapter | Target and horizon | Row meaning | Missing values | Important limitations |
| --- | --- | --- | --- | --- |
| `german_credit` | Bad credit-risk label; the source does not specify a reliable prediction horizon | Historical credit application | The source file has no blank fields; the pipeline still validates and imputes introduced missing values | Small historical sample, encoded categories, and an ambiguous cost context |
| `give_me_some_credit` | Serious delinquency of 90+ days within two years | Applicant record | `MonthlyIncome` and `NumberOfDependents` may be missing and are imputed inside the fitted pipeline | Competition provenance and selection limits; age and other fields may be proxy variables |

Both adapters retain their documented audit attributes outside model features. The
adapter metadata records the source URL, license, citation, target definition,
prediction horizon, missing-value behavior, protected-attribute notes, and known
limitations in every alternate training artifact.
