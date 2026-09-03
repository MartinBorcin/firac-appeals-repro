# Configuration Index

Generic portable examples use `${OPENAI_BASE_URL}`, `${OPENAI_API_KEY}`, and
`${FIRAC_MODEL}`. Exact model configurations preserve reported model IDs,
sampling parameters, seeds, token limits, and concurrency settings; endpoint
locations and credentials remain environment supplied.

## Annotation

- `annotation/mock.yaml`: credential-free smoke test.
- `annotation/firac_pipeline.yaml`: portable staged FIRAC pipeline.
- `annotation/local_vllm_qwen.yaml`: high-throughput Qwen3.6 configuration used
  for the release, exposed through portable endpoint environment variables.

## Downstream

- `downstream_qwen.yaml`, `downstream_qwen3_32b.yaml`;
- `downstream_mistral_small_3_2_24b.yaml` and binary/authority variants;
- `downstream_saul_7b*.yaml`;
- `downstream_saullm_54b*.yaml`.

The v2 binary prompt is mandatory for reported LJP runs. Superseded ternary
prompts and outputs are omitted from this release.

## Quality evaluation

- `llm_rubric_gpt_oss_120b_public_v4.yaml`: full-corpus GPT-OSS screen;
- `label_studio/firac_eval_round2.xml`: exact human interface;
- `serving/vllm_models.yaml`: portable record of the model-serving profiles.

The serving record retains model, accelerator, parallelism, chat-template,
context, and batching choices while omitting Slurm queues, ports, cache paths,
and other site-specific launch plumbing. Any compatible endpoint may be used.
