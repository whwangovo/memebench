"""MemeBench pipeline scripts.

Organized into categories:

1. Collection (pipelines/collection/)
   - collect_reddit.py: Scrape top memes from Reddit subreddits
   - prepare_english.py: Integrate selected memes into dataset

2. Annotation (pipelines/annotation/)
   - annotate_memes.py: Generate ground truth from Excel + images

3. Inference (pipelines/inference/)
   - vanilla.py: Vanilla VLM response
   - cot.py: Chain-of-Thought variant
   - search_cot.py: Search-augmented CoT (KAR w/o CultureBase)
   - kar.py: KAR (CultureBase + web search)
   - oracle_progressive.py: Progressive oracle ablation

4. Evaluation (pipelines/evaluation/)
   - judge.py: LLM judge evaluation
   - score.py: Calculate metrics
   - compare.py: Cross-variant comparison
   - compare_split.py: ZH/EN split comparison

Usage (run from project root):
    python pipelines/annotation/annotate_memes.py
    python pipelines/inference/vanilla.py
    python pipelines/evaluation/judge.py --candidate <file>
    python pipelines/evaluation/compare.py
"""

__all__ = [
    "collection",
    "annotation",
    "inference",
    "evaluation",
]
