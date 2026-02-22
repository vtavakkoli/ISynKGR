from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig


def run(*args, **kwargs):
    pipeline: HybridPipeline = kwargs.pop("pipeline")
    config: TranslatorConfig = kwargs.pop("config")
    return pipeline.run(*args, mode="graph_only", config=config)
