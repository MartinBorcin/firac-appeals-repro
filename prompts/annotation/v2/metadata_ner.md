You are extracting metadata from an Indian appellate judgment.

Return only one valid JSON object. Do not include markdown or explanations.
Use empty strings or empty lists when a value is not present.

Schema:

    {
      "neutral_citation": "",
      "court": "",
      "district": "",
      "impugned_court": "",
      "bench_coram": [],
      "judgment_by": [],
      "judgment_date": "",
      "lawyers": {
        "for_petitioner_or_appellant": [],
        "for_respondent": []
      },
      "case_details": [
        {
          "case_no": [],
          "petitioner_or_appellant": [],
          "respondent": []
        }
      ]
    }

Rules:
- Extract from the judgment itself only.
- Normalize whitespace and preserve party/case-number wording where useful.
- Judgment date should be dd/mm/yyyy when unambiguous, otherwise copy the date as written.
- Do not infer missing names from outside knowledge.

Example output:

    {
      "neutral_citation": "2024 INSC 123",
      "court": "Supreme Court of India",
      "district": "",
      "impugned_court": "High Court of Delhi",
      "bench_coram": ["A. B. Judge", "C. D. Judge"],
      "judgment_by": ["A. B. Judge"],
      "judgment_date": "14/03/2024",
      "lawyers": {
        "for_petitioner_or_appellant": ["Name One"],
        "for_respondent": ["Name Two"]
      },
      "case_details": [
        {
          "case_no": ["Civil Appeal No. 123 of 2024"],
          "petitioner_or_appellant": ["Example Appellant"],
          "respondent": ["Example Respondent"]
        }
      ]
    }

Judgment:
<judgment>
{judgment}
</judgment>
