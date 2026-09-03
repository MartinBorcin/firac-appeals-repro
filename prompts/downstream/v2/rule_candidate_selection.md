You are given case facts, predicted legal issues, and a closed list of candidate authorities retrieved from prior training cases. Select and rank up to 10 candidates that are most relevant to resolving the case.

Use only authorities from the supplied candidate list. Copy each selected candidate's type, reference, provision, case_name, and citation exactly. Give a short relevance explanation based only on the supplied facts and issues. Do not introduce outside authorities.

Return only valid JSON:

    {
      "rules": [
        {
          "type": "statute|rule|regulation|constitutional_provision|precedent|legal_principle|other",
          "reference": "candidate reference",
          "provision": "candidate provision or empty string",
          "case_name": "candidate case name or empty string",
          "citation": "candidate citation or empty string",
          "relevance": "brief relevance"
        }
      ]
    }

Case and candidates:
{input_text}
