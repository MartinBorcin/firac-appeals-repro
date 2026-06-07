You are extracting the logical reasoning process from an Indian appellate
judgment in a clear, issue-by-issue form.

Return only one valid JSON object. Do not include markdown or explanations.

Schema:

    {
      "analysis": [
        {
          "issue": "",
          "steps": [],
          "conclusion": ""
        }
      ]
    }

Goal:
Capture the logical process by which the rules apply to the facts to answer each
decisive issue. Write the steps in present tense as an outline of the reasoning
itself, not as a report about what the judge did. Use clear steps, but do not
force the reasoning into rigid syllogisms if the judgment is more practical,
discretionary, or multi-factorial.

Rules:
- Structure by issue.
- "steps" should be an ordered list of short reasoning statements. This list can be long, in case of a long logical process or consideration of many perspectives.
- Each step should connect facts, rules, evidence, procedure, or precedent to
  the court's reasoning.
- Keep legal reasoning separate from party submissions.
- Do not add reasoning that is not present in the judgment.
- The issue-level conclusion should follow from the steps and should not be more
  specific than the judgment supports.
- Use present tense and avoid reported-speech openings such as "The court first
  identified", "It considered", or "The judge found".
- If a formal syllogistic structure is natural, use it within the steps, but it
  is not required.

Example output:

    {
      "analysis": [
        {
          "issue": "Whether the High Court was justified in condoning the delay?",
          "steps": [
            "The appeal is filed after the prescribed limitation period.",
            "Section 5 of the Limitation Act requires sufficient cause for delay.",
            "The explanation describes administrative movement of files but does not account for the full period of delay.",
            "An incomplete explanation is insufficient because delay condonation is discretionary and fact-dependent."
          ],
          "conclusion": "The explanation did not establish sufficient cause for condonation."
        }
      ]
    }

Judgment:
<judgment>
{judgment}
</judgment>

Decisive issues:
<issues>
{issues}
</issues>

Rules and authorities:
<rules>
{rules}
</rules>
