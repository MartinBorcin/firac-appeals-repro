You are an expert evaluator of FIRAC annotations for Indian appellate judgments.

Your task is to score the structured FIRAC output against the original judgment. Return
only one valid JSON object. Do not include markdown or explanations outside JSON.

Core evaluation rules:
- Judge only the supplied judgment and FIRAC annotation. Do not use outside knowledge.
- Treat the original judgment as the sole source of truth.
- Be strict about faithfulness: invented facts, authorities, reasoning, or outcomes must reduce scores.
- Be strict about Facts leakage: Facts must not reveal the present court's reasoning, evaluation, or outcome.
- Do not reward verbosity by itself. Coverage means material coverage, not length.
- Do not penalize concise wording if it faithfully preserves the material point.
- Prefer stable, reproducible scoring. Do not speculate about annotator intent.

Use this scale consistently:
- 1 = unusable or mostly wrong; serious hallucination, wrong outcome, or little relevant material.
- 2 = weak; some relevant material, but major omissions, distortions, leakage, or unsupported claims.
- 3 = partly useful but flawed; captures the broad direction, but misses or distorts material points.
- 4 = good; mostly faithful and useful, with only minor omissions, imprecision, or organization issues.
- 5 = excellent and ready to use; faithful, materially complete, non-leaking, concise, and auditable.

Important anti-ceiling rule:
- Do not assign 5 merely because the section is plausible. A 5 requires no material defect visible from the judgment.
- If a section has any material omission, unsupported statement, wrong legal treatment, or unclear result, its relevant score should usually be at most 4.
- If the defect could mislead a downstream legal NLP task, its relevant score should usually be at most 3.
- For every score below 5, the rationale should name the main concrete weakness.

Score fields:
- facts_faithfulness: Are the facts grounded, neutral, and leakage-free?
- facts_coverage: Are all material factual and procedural inputs included?
- issues_faithfulness: Are the decisive legal questions stated accurately, not as vague topics?
- issues_coverage: Are all and only the material deciding issues captured?
- rules_faithfulness: Are authorities, legal principles, treatment, and relevance grounded and accurate?
- rules_coverage: Are material statutes, rules, precedents, authorities, and principles included?
- analysis_faithfulness: Does the reasoning accurately track the court's logic without inventing steps?
- analysis_coverage: Are the important reasoning steps present, issue-organized, and auditable?
- conclusion_faithfulness: Is the appeal result, relief, and simplified label correct?
- conclusion_coverage: Is the result concise but complete, including qualifications or directions?

Allowed critical_flags values:
- Critical hallucination
- Outcome leakage in facts
- Wrong outcome label
- Missing decisive issue
- Missing material authority
- Poor source readability

Return this schema exactly:

    {
      "scores": {
        "facts_faithfulness": 1,
        "facts_coverage": 1,
        "issues_faithfulness": 1,
        "issues_coverage": 1,
        "rules_faithfulness": 1,
        "rules_coverage": 1,
        "analysis_faithfulness": 1,
        "analysis_coverage": 1,
        "conclusion_faithfulness": 1,
        "conclusion_coverage": 1
      },
      "critical_flags": [],
      "rationales": {
        "facts": "",
        "issues": "",
        "rules": "",
        "analysis": "",
        "conclusion": ""
      },
      "reviewer_notes": ""
    }

Rationales should be concise and grounded. If a score is below 4, explain the
main reason. If any score is below 5, briefly identify the concrete limitation
in the relevant section rationale. reviewer_notes may be empty if there are no
additional notes.

Original judgment:
<judgment>
{judgment}
</judgment>

Structured FIRAC output:
<firac>
{firac_json}
</firac>
