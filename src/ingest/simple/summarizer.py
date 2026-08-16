def create_prompt(item_description: str, max_chars: int = 512) -> str:
    return (
        "### You are a salesperson helping recommend items to people. Your objective is "
        "to identify the most relevant buyer-facing details in the provided product "
        "description and summarize them for recommendation matching.\n\n"
        "# Focus on concrete attributes such as item type, intended use, compatible "
        "instruments or gear, materials, size, features, included accessories, and "
        "any clear audience or skill-level fit. Do not add facts, claims, or opinions "
        "that are not present in the description.\n\n"
        f"### Keep the summary under {max_chars} characters.\n\n"
        f"### Product description:\n{item_description}"
    )
