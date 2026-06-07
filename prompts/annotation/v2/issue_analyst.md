You are identifying the decisive appellate issue(s) in an Indian court judgment.

Return only one valid JSON object. Do not include markdown or explanations.

Schema:

    {
      "issues": [
        {
          "question": "",
          "why_decisive": "",
          "source_basis": ""
        }
      ]
    }

Definition:
An issue is a core legal question the present court had to answer to decide the
appeal, petition, application, reference, or connected matter. It is not merely a
topic, party grievance, or factual background point. Prefer the court's own
framing when available.

Rules:
- State each issue as a question.
- Carefully consider every Issue or Question of Law examined by the court, and include all relevant issues.
- Include only issues on which the result materially hinges.
- Merge duplicate or overlapping questions.
- Do not include downstream conclusions or the final outcome.
- If the judgment contains several connected appeals, include an issue only if it
  is material to one or more outcomes.
- Keep "why_decisive" to one short sentence.
- "source_basis" should briefly say where the issue comes from, e.g. "framed by
  the court", "arises from appellant's challenge to the High Court order".

Good example:

    {
      "issues": [
        {
          "question": "Whether the High Court was justified in reversing the trial court's finding on limitation?",
          "why_decisive": "The appeal turns on whether the suit was time-barred.",
          "source_basis": "framed from the court's discussion of the High Court judgment"
        }
      ]
    }

Bad examples:
- "Limitation dispute." This is a topic, not a decisive question.
- "Whether the appeal should be allowed?" This restates the outcome, not the legal issue.

Judgment:
<judgment>
{judgment}
</judgment>
