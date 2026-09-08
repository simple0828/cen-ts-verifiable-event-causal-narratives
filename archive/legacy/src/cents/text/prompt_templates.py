EVENT_EXTRACTION_PROMPT = """You are an expert in time series forecasting and event-causal reasoning.

Given a timestamped text snippet paired with a numerical time series dataset, extract structured events that may plausibly affect future numerical variables.

Return ONLY valid JSON lines. Do not include markdown.

For each event, return:
{
"time": "",
"event_phrase": "",
"event_type": "<policy|market|weather|health|traffic|energy|agriculture|sports|social|other>",
"entities": ["", ""],
"target_variable": "",
"polarity": "<positive|negative|neutral|uncertain>",
"expected_lag_min": ,
"expected_lag_max": ,
"magnitude": "<weak|medium|strong|uncertain>",
"confidence": <float between 0 and 1>,
"rationale": ""
}

Rules:

1. Extract only events that could plausibly affect the numerical target or covariates.
2. Avoid generic background statements.
3. If the text contains no useful event, return an empty list.
4. Be conservative. Use low confidence for ambiguous events.
5. Do not claim causality unless the text suggests a plausible temporal mechanism.
6. Lag is measured in dataset time steps, not calendar days, unless the data frequency is known.

Input:
Dataset domain: {domain}
Target variable: {target_variable}
Available numerical variables: {variables}
Timestamp: {timestamp}
Text:
{text}
"""

