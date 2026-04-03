from isynkgr.pipeline.adaptive_candidate_ranker import HybridPipeline, TranslatorConfig


def run(*args, **kwargs):
    pipeline: HybridPipeline = kwargs.pop("pipeline")
    config: TranslatorConfig = kwargs.pop("config")
    return pipeline.run(*args, mode="rule_only", config=config)
