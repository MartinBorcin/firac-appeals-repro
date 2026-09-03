You are evaluating a FIRAC annotation of an Indian appellate judgment. Use only the supplied judgment as the source of truth. Do not reward plausible legal background that is not present in the judgment. Score each field independently and return one JSON object with no other text.

Use these exact human-review questions:
- Facts faithfulness/correctness: Are the facts grounded, neutral, and leakage-free?
- Facts coverage/usefulness: Are all material factual and procedural inputs included?
- Issues faithfulness/correctness: Are the decisive legal questions stated accurately?
- Issues coverage/usefulness: Are all and only the material deciding issues captured?
- Rules/Authorities faithfulness/correctness: Are authorities, legal principles, treatment, and relevance grounded and accurate? For non-precedents, `Not applicable` treatment means treatment type does not apply, not that the authority is irrelevant.
- Rules/Authorities coverage/usefulness: Are material statutes, rules, precedents, authorities, and legal principles included?
- Analysis faithfulness/correctness: Does the reasoning accurately track the court's logic?
- Analysis coverage/usefulness: Are the important reasoning steps present and auditable?
- Conclusion faithfulness/correctness: Is the appeal result, relief, and simplified label correct? The simplified label is coarse: partly allowed maps to allowed.
- Conclusion coverage/usefulness: Is the result concise but complete, including qualifications or directions?

General scoring anchors:
- 1: unusable or mostly wrong.
- 2: serious problems; useful only with substantial correction.
- 3: partly useful but incomplete, vague, or containing noticeable errors.
- 4: good; minor omissions or wording issues.
- 5: excellent; faithful, complete for purpose, clear, and ready to use.

Facts faithfulness/correctness:
- 1: fabricates facts, reverses parties/events, or leaks the present appeal outcome throughout.
- 2: contains multiple unsupported details or substantial outcome leakage.
- 3: mostly grounded but includes some unsupported details, legal reasoning, or avoidable outcome hints.
- 4: grounded and neutral with only minor wording or support issues.
- 5: fully grounded, neutral, and leakage-free; no added material facts.

Facts coverage/usefulness:
- 1: misses most material facts and procedural history.
- 2: captures only a small portion of the case background.
- 3: captures the central background but misses some claims, procedural steps, or material context.
- 4: covers nearly all important factual and procedural inputs.
- 5: comprehensively enumerates material facts, claims, procedural history, and party positions needed to understand the appeal.

Issues faithfulness/correctness:
- 1: identifies questions unrelated to the judgment.
- 2: misstates the central legal question or frames it at the wrong procedural level.
- 3: partly captures the issue but is broad, vague, or mixes issues with facts and conclusions.
- 4: accurately captures the main deciding question with minor imprecision.
- 5: precisely states the decisive legal question or questions in the court's terms.

Issues coverage/usefulness:
- 1: omits the central issue.
- 2: includes only peripheral issues.
- 3: includes the main issue but misses another material issue or sub-issue.
- 4: covers the decisive issues with minor omissions.
- 5: covers all and only the material deciding issues at the right level of granularity.

Rules/Authorities faithfulness/correctness:
- 1: fabricates authorities or attributes rules to the court that are not in the judgment.
- 2: includes multiple wrong citations, wrong treatments, or misleading relevance explanations.
- 3: mostly grounded but with some incorrect treatment, overbroad rule statements, or unsupported authorities.
- 4: accurate authorities and treatments with only minor omissions or imprecision.
- 5: all listed authorities/principles are grounded, accurately described, and treatment/relevance is correct.

Rules/Authorities coverage/usefulness:
- 1: misses nearly all legal authorities and principles.
- 2: captures a few authorities but misses most material rules.
- 3: captures the main rule/authority but omits important supporting authorities, treatments, or relevance.
- 4: covers nearly all material rules, authorities, and legal principles.
- 5: comprehensively covers all material statutes, rules, precedents, principles, treatment types, and relevance needed to audit the reasoning.

Analysis faithfulness/correctness:
- 1: invents reasoning or contradicts the judgment's logic.
- 2: contains major reasoning errors, missing links, or unsupported inferences.
- 3: broadly follows the judgment but has gaps, vague steps, or occasional unsupported reasoning.
- 4: accurately tracks the reasoning with minor simplifications.
- 5: faithfully and clearly reconstructs the court's logical process from issue to conclusion.

Analysis coverage/usefulness:
- 1: gives little or no reasoning.
- 2: states the outcome but misses most intermediate reasoning.
- 3: captures the main reasoning path but omits important steps or issue-level structure.
- 4: covers nearly all important reasoning steps.
- 5: complete, issue-organized, auditable reasoning that would support downstream explanation or summarization.

Conclusion faithfulness/correctness:
- 1: wrong outcome.
- 2: partly wrong outcome or seriously misleading relief.
- 3: correct broad outcome but missing or misstating important relief details.
- 4: correct outcome and relief with minor wording issues.
- 5: exactly captures the appeal result, relief/order, and simplified label.

Conclusion coverage/usefulness:
- 1: no usable conclusion.
- 2: gives only a vague result.
- 3: gives the broad result but omits important qualifications, partial allowance/dismissal, remand, costs, or directions.
- 4: includes nearly all material result details.
- 5: concise, complete, and directly follows from the analysis.

Scoring controls:
- A plausible section is not automatically a 5.
- Do not force all scores for a case to be similar.
- Any material unsupported statement, omission, wrong treatment, leakage, or unclear relief normally prevents a 5.
- A defect capable of misleading a downstream legal task normally caps the relevant score at 3.
- Concise paraphrase is acceptable when it preserves the material legal content.
- Do not infer missing content from outside legal knowledge.

Allowed critical flags:
- Critical hallucination
- Outcome leakage in facts
- Wrong outcome label
- Missing decisive issue
- Missing material authority
- Poor source readability

Return exactly:
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
    "facts": "concrete defects or empty when none",
    "issues": "concrete defects or empty when none",
    "rules": "concrete defects or empty when none",
    "analysis": "concrete defects or empty when none",
    "conclusion": "concrete defects or empty when none"
  },
  "reviewer_notes": ""
}

Original judgment:
<judgment>
{judgment}
</judgment>

Structured FIRAC output:
<firac>
{firac_json}
</firac>
