You are creating a structured FIRAC annotation for an Indian appellate judgment.

Return only one valid JSON object. Do not include markdown or explanations.

Schema:

    {
      "facts": [],
      "issues": [],
      "rules": [],
      "analysis": [],
      "conclusion": {
        "label": "",
        "text": "",
        "simplified_label": "allowed|dismissed|other"
      }
    }

Role:
You are the final annotator. You have access to the judgment and to outputs from
specialized sub-experts, but you are not bound by them. You may correct, merge,
discard, rewrite, or add to any sub-expert output when the judgment supports it.
Your purpose is to produce one harmonized, faithful FIRAC annotation.

Field requirements:

facts:
- A list of neutral factual sentences.
- Enumerate all material facts known before the present court's ruling.
- Include procedural history, lower-court decisions, claims, defences, evidence,
  contracts, statutory notices, deadlines, and relief sought when material.
- Facts must be leakage-free: do not hint at the present appeal's outcome, the
  present judge's reasoning, or the present court's final findings.
- Do not use phrases like "the Court held", "the appeal was dismissed", "the
  argument had no merit", or "the petitioner failed to prove" unless describing
  a lower court or a party's submission and making that clear.
- Do not add details that are not explicitly present in the judgment.
- Facts should include all information that is needed to fully reconstruct the judge's reasoning process. 

issues:
- A list of decisive legal questions.
- Each issue should be a question the present court had to answer and on which
  the result hinged.

rules:
- A list of structured rule objects copied or consolidated from the legal
  researcher.
- Treat "rules" broadly as rules/authorities/principles, not only legal rules in
  the narrow doctrinal sense.
- Include statutes, rules, regulations, constitutional provisions, precedents,
  legal principles, doctrines, tests, standards, and "other" legal materials
  where relevant.
- Each rule should explain briefly why the present court considered it.

analysis:
- A list of issue-level objects with "issue", "steps", and "conclusion".
- Steps should be clear and ordered but not artificially over-formalized.
- Write steps in present tense as an outline of the logical reasoning process,
  not as reported speech about the judge.
- Avoid openings such as "The court finds", "The court considers", or "It holds"
  unless unavoidable for clarity.
- The analysis should express legal reasoning, not party argument.

conclusion:
- "label" is the best canonical legal disposition, e.g. "Appeal allowed",
  "Appeal dismissed", "Appeal partly allowed", "Petition dismissed", "Matter
  remitted to High Court", "Order accordingly".
- "text" is one concise sentence stating what the present court decided,
  including important remand/direction/partial-relief details.
- "simplified_label" is:
  - "allowed" when the appeal/petition/application is allowed or partly allowed;
  - "dismissed" when it is dismissed;
  - "other" when allowed/dismissed is not applicable, unclear, mixed across
    multiple matters, only a reference is answered, or no final disposition is made.

Example output:

    {
      "facts": [
        "The appellant challenged an order of the High Court reversing the trial court's decision.",
        "The dispute concerned whether the suit was filed within the prescribed limitation period.",
        "The appellant argued that the delay should be condoned because the relevant documents moved through several administrative offices.",
        "The respondent opposed condonation and argued that the explanation did not cover the entire period of delay."
      ],
      "issues": [
        "Whether the High Court was justified in refusing to condone the delay in filing the appeal?"
      ],
      "rules": [
        {
          "type": "statute",
          "reference": "Limitation Act, 1963",
          "provision": "Section 5",
          "case_name": "",
          "citation": "",
          "treatment_type": "Not applicable to authority type",
          "relevance": "The court considered whether the appellant had shown sufficient cause for delay.",
          "used_by_court": true
        }
      ],
      "analysis": [
        {
          "issue": "Whether the High Court was justified in refusing to condone the delay in filing the appeal?",
          "steps": [
            "Section 5 requires a sufficient and credible explanation for the delay.",
            "The explanation offered by the appellant must account for the full period of delay.",
            "The explanation does not address the full delay and lacks necessary particulars."
          ],
          "conclusion": "The appellant did not establish sufficient cause for condonation."
        }
      ],
      "conclusion": {
        "label": "Appeal dismissed",
        "text": "The appeal was dismissed and the High Court's refusal to condone delay was affirmed.",
        "simplified_label": "dismissed"
      }
    }

Judgment:
<judgment>
{judgment}
</judgment>

Issue analyst output:
<issues>
{issue_analysis}
</issues>

Legal researcher output:
<researcher>
{researcher_analysis}
</researcher>

Reasoning extractor output:
<logician>
{logic_analysis}
</logician>
