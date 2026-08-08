def run_system(system_model: str):
    if system_model == "item_description_ranker":
        from llm_knowledge_enhancement.system.item_description_ranker import run

        run()
