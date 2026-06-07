You are verifying a FIRAC annotation against the original Indian appellate
judgment.

Return only one valid JSON object. Do not include markdown or explanations.

Schema:

    {
      "is_usable": true,
      "facts_leakage": {
        "has_leakage": false,
        "problem_sentences": [],
        "explanation": ""
      },
      "faithfulness_errors": [],
      "missing_material_points": [],
      "rules_issues": [],
      "analysis_issues": [],
      "conclusion_issue": "",
      "recommended_quality_flags": []
    }

Check:
- Facts contain only information known before the present court's ruling.
- Facts do not reveal the present appeal's outcome or the present judge's
  reasoning.
- Issues are decisive legal questions, not vague topics.
- Rules are present in the judgment and relevant to the court's reasoning.
- Analysis follows the judgment's reasoning and does not invent steps.
- Conclusion label/text matches the actual disposition.

Example output:

    {
      "is_usable": false,
      "facts_leakage": {
        "has_leakage": true,
        "problem_sentences": ["The High Court correctly dismissed the appellant's claim."],
        "explanation": "The sentence reveals an evaluative finding rather than pre-ruling facts."
      },
      "faithfulness_errors": [],
      "missing_material_points": ["The annotation omits the respondent's limitation objection."],
      "rules_issues": [],
      "analysis_issues": ["The analysis attributes a finding to the court that is not stated in the judgment."],
      "conclusion_issue": "",
      "recommended_quality_flags": ["possible_facts_leakage", "missing_material_facts"]
    }

Judgment:
<judgment>
{judgment}
</judgment>

FIRAC annotation:
<firac>
{firac_json}
</firac>
