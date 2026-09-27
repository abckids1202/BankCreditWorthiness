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
