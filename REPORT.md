# Churn Risk Predictor — Product Manager Brief

**To:** Product & Customer Success Leadership  
**From:** ML Team  
**Date:** July 2026  
**Re:** Who is about to churn, and what to do about it

---

## What We Built and Why

We lose customers every quarter. Until now, the CS team has been reacting — finding out a customer churned only after they cancelled. This tool flips that around. It scores every active account every week and tells you who is likely to cancel in the next 90 days, ranked by risk level, with a plain-English explanation of why.

The output is not a spreadsheet full of numbers. It is a prioritised call list with talking points.

---

## How We Defined "Churn"

This matters more than the model itself. We defined a churned customer as one who cancels or fails to renew within 90 days of the prediction date. We chose 90 days because it gives the CS team enough runway to intervene — 30 days is too short to change behaviour, and 180 days produces too many false alarms.

We score customers at the start of each week using data available at that moment. We are careful not to include anything that would only be known after a cancellation decision has already been made (for example, we do not use a "cancellation requested" flag as an input — that would be cheating and would be useless in practice).

---

## What Signals Predict Churn

We looked at 16 signals across five categories. The three that matter most, in order:

**1. Login activity.** An account that has fewer than 5 logins in the past 30 days, or whose last login was more than 30 days ago, is significantly more likely to churn. Disengagement is almost always the first sign something is wrong.

**2. Feature adoption.** Customers using less than 20% of the product's features have much lower switching costs. They have not built the product into their workflow, so leaving is easy.

**3. Billing friction.** Even a single billing failure in the past six months roughly doubles churn risk. It signals either financial stress or a relationship that is already de-prioritised internally.

Secondary signals — high support ticket volume, a contract renewal within 60 days, low or missing NPS score, and fewer than two active integrations — all add meaningful lift to the model.

---

## How Accurate Is It

The model correctly identifies 40% of churners before they leave, while keeping false alarms manageable (about 1 in 4 accounts flagged as high-risk actually churn). On a standard ML benchmark (ROC-AUC), it scores 0.66 — meaningful, but not perfect.

To be direct: this tool will miss some churners, and it will flag some accounts that stay. That is normal and expected. The goal is not perfection — it is to give the CS team a better starting point than gut feel or last-month's revenue report.

---

## What the CS Team Should Do

For each **🔴 HIGH risk** account: call within the week. The tool shows exactly which signals triggered the flag, so the account manager can open with something specific — "I noticed logins have dropped off — is there something blocking the team?" rather than a generic check-in.

For **🟡 MEDIUM risk** accounts: schedule a check-in within two weeks and share a relevant feature tutorial or success story.

For **🟢 LOW risk** accounts: routine quarterly review. These are also the best candidates for upsell conversations.

---

## Assumptions and Data Sources

All data is synthetic and generated for demonstration purposes. In production, this model would be trained on your CRM export, product analytics (login events, feature flags), billing system, and support ticket counts. The model would need retraining every quarter as customer behaviour evolves.

The NPS and CS health score columns had meaningful missing rates (20% and 15% respectively). We treated missingness itself as a signal — an account with no NPS on record is more likely to churn than the median, which the model captures.

---

## Next Steps

1. Connect the tool to live CRM and product analytics data
2. Run a 30-day pilot with two CS managers to validate that flagged accounts respond differently to intervention
3. Track intervention outcomes to build a labelled dataset for model improvement
4. Add a Slack integration to push the weekly high-risk list automatically
