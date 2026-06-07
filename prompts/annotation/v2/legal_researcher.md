You are extracting the legal rules and authorities the present court considered
while deciding an Indian appellate judgment.

Return only one valid JSON object. Do not include markdown or explanations.

Schema:

    {
      "rules": [
        {
          "type": "statute|rule|regulation|constitutional_provision|precedent|legal_principle|other",
          "reference": "",
          "provision": "",
          "case_name": "",
          "citation": "",
          "treatment_type": "Relied on|Referred|Distinguished|Overruled|Not applicable to authority type",
          "relevance": "",
          "used_by_court": true
        }
      ]
    }

Definitions:
- "reference" is the statute/rule/precedent/principle as it appears or is clearly
  identified in the judgment.
- "provision" is the section, article, rule, order, regulation, or clause number.
- "case_name" and "citation" are used for precedents only.
- "treatment_type" applies to precedents. Use "Not applicable to authority type"
  for non-precedents; this means precedent treatment does not apply to that
  authority type, not that the authority is legally irrelevant.
- "relevance" is a very short explanation of why the court considered it.
- "used_by_court" is true when the judge/court uses the rule in reasoning. It is
  false when only a party cited it and the court did not adopt, reject, or analyze it.

Rules:
- Include statutes, rules, regulations, constitutional provisions, precedents,
  legal doctrines, tests, standards, and other legal materials that matter to the
  court's reasoning.
- Treat "rules" broadly as rules/authorities/principles, not only black-letter
  legal rules in the narrow sense.
- Do not invent citations, provisions, or principles.
- Do not include authorities found only inside quoted text unless the present
  court itself discusses or adopts them.
- Prefer completeness over artificial filtering, but mark party-only citations as
  "used_by_court": false.
- Keep "relevance" concise, usually one sentence.

Example output:

    {
      "rules": [
        {
          "type": "statute",
          "reference": "Limitation Act, 1963",
          "provision": "Section 5",
          "case_name": "",
          "citation": "",
          "treatment_type": "Not applicable to authority type",
          "relevance": "The court considered whether delay could be condoned on sufficient cause.",
          "used_by_court": true
        },
        {
          "type": "precedent",
          "reference": "Collector, Land Acquisition v. Katiji",
          "provision": "",
          "case_name": "Collector, Land Acquisition v. Katiji",
          "citation": "(1987) 2 SCC 107",
          "treatment_type": "Relied on",
          "relevance": "The court used it for the principle that delay condonation should be considered pragmatically.",
          "used_by_court": true
        },
        {
          "type": "legal_principle",
          "reference": "Appellate interference with concurrent findings is limited",
          "provision": "",
          "case_name": "",
          "citation": "",
          "treatment_type": "Not applicable to authority type",
          "relevance": "The court considered the standard for disturbing findings of fact.",
          "used_by_court": true
        }
      ]
    }

Judgment:
<judgment>
{judgment}
</judgment>
