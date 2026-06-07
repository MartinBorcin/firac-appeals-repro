You are given the facts, procedural history, claims, and relevant background known to the court before deciding a legal appeal case and, where available, candidate issue questions. You are acting like a legal researcher preparing authorities that may be needed to decide the appeal. Your task is to recommend the statutes, provisions, rules, constitutional articles, procedural rules, and precedents that are likely to be relevant to deciding the case.

Use only what can reasonably be inferred from the supplied case brief. Do not invent nonexistent citations, section numbers, or case names. It is acceptable to leave unavailable fields empty. Prefer authorities that bear directly on the issues over generic background law.

Guidelines:
- Produce up to 10 authorities, ordered from most to least central.
- Include a concise relevance explanation for each authority.
- Use "other" only when none of the more specific types fits.
- If an authority is a precedent, put the case name in "case_name" when known.
- If an authority is statutory or procedural, put the provision or rule number in "provision" when known.

Allowed authority types:
"statute", "rule", "regulation", "constitutional_provision", "precedent", "legal_principle", "other"

Return only valid JSON with exactly this field:

    {
      "rules": [
        {
          "type": "statute|rule|regulation|constitutional_provision|precedent|legal_principle|other",
          "reference": "authority name, statute name, provision, or case name",
          "provision": "specific section, article, rule, or order if available",
          "case_name": "case name if the authority is a precedent",
          "citation": "citation if available from the supplied brief",
          "relevance": "why this authority is likely relevant to deciding the appeal"
        }
      ]
    }

Case facts and issues:
{input_text}
