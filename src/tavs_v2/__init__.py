def __getattr__(name):
    if name in ("TavsEspStrategy", "TavsEspConfig"):
        from src.tavs_v2.tavs_esp_strategy import TavsEspStrategy, TavsEspConfig
        return locals()[name]
    if name in ("TAVSESPPipeline", "PipelineConfig", "PipelineResults", "create_example_configs"):
        from src.tavs_v2.end_to_end_pipeline import (
            TAVSESPPipeline, PipelineConfig, PipelineResults, create_example_configs,
        )
        return locals()[name]
    raise AttributeError(f"module 'src.tavs_v2' has no attribute {name!r}")
