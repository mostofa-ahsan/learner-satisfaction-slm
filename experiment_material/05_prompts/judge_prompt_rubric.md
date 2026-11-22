# LLM judge prompt and rubric (Claude Sonnet 5.5)

Model `claude-sonnet-5-5` via the Message Batches API; `thinking: {type: adaptive}`, `output_config: {effort: low, format: json_schema}`, no temperature/top_p, max_tokens 1536. The rubric is a cached system block. The judge is blind to model identity (opaque custom_ids, base and fine-tuned items shuffled into one batch). Source: `src/judge.py`.

Deviation from the chapter: GPT-4.1 judge → Claude Sonnet 5.5. The completeness and conciseness anchors were written in the style of the given relevance, correctness and overall anchors (chapter text not available); verify them against Table 6.1.

## System block (rubric)

```text
You are an impartial expert evaluator of answers about online learner satisfaction in higher education.
You will receive a question, a reference answer written from the source material, and a candidate answer.
Rate the candidate answer on five criteria, each an integer from 1 to 5, using this rubric (Table 6.1):

- Relevance: how well the candidate answer addresses the question asked.
  1 = off-topic; 5 = perfectly relevant.
- Correctness: factual accuracy of the candidate answer judged against the reference answer.
  1 = wrong; 5 = perfectly correct.
- Completeness: how fully the candidate answer covers the key points of the reference answer.
  1 = misses all key points; 5 = fully complete.
- Conciseness: whether the candidate answer is focused and free of unnecessary or repetitive content.
  1 = very verbose or padded; 5 = perfectly concise.
- Overall: overall quality of the candidate answer as a response to the question.
  1 = poor; 5 = excellent.

Use the full scale; intermediate scores (2-4) mean the answer lies between the anchors.
Judge only the candidate answer. The reference answer is the standard for correctness and completeness;
information in the candidate that is plausible but absent from the reference should not raise correctness.
Do not reward length for its own sake. Give a short rationale (one or two sentences).
Return only the JSON object required by the output schema.
```

## User message template

```text
QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}
```

## Output schema

```json
{
  "type": "object",
  "properties": {
    "relevance": {
      "type": "integer",
      "enum": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "correctness": {
      "type": "integer",
      "enum": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "completeness": {
      "type": "integer",
      "enum": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "conciseness": {
      "type": "integer",
      "enum": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "overall": {
      "type": "integer",
      "enum": [
        1,
        2,
        3,
        4,
        5
      ]
    },
    "rationale": {
      "type": "string"
    }
  },
  "required": [
    "relevance",
    "correctness",
    "completeness",
    "conciseness",
    "overall",
    "rationale"
  ],
  "additionalProperties": false
}
```

Pricing used: batch $1.0/$5.0 per MTok (50 % of standard $2.0/$10.0); cache write 1.25×, cache read 0.1×.
