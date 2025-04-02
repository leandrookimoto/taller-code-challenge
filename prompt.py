from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

examples = [
            {
                "logs": """
                [2025-02-20 14:32:10] INFO - Agent Response: Hi
                [2025-02-20 14:33:15] INFO - Agent Response: Hi
                [2025-02-20 14:34:02] INFO - Agent Response: Sorry
                """,
                "answer": """
                    Log Summary:
                    - INFO messages: 3
                    - ERROR messages: 0
                    - WARNING messages: 0

                    Top 3 AI Responses:
                    1. "Hi" (2 times)
                    2. "Sorry" (1 time)

                    Most Common Errors:
                    No errors.
        """,
            },
            {
                "logs": """
                [2025-02-20 14:32:10] INFO - Agent Response: "Hello! How can I help you today?"
                [2025-02-20 14:33:15] ERROR - Model Timeout after 5000ms
                [2025-02-20 14:34:02] INFO - Agent Response: "I'm sorry, I didn't understand that."
                """,
                "answer": """
                Log Summary:
                - INFO messages: 2
                - ERROR messages: 1
                - WARNING messages: 0

                Top 3 AI Responses:
                1. "Hello! How can I help you today?" (1 times)
                2. "I'm sorry, I didn't understand that." (1 times)
                Most Common Errors:
                1. Model Timeout after 5000ms (1 time)
                
        """
            }
        ]
    
example_prompt = PromptTemplate(
input_variables=["logs", "answer"], template="Logs: {logs}\n{answer}"
        )

prompt_few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Logs: {logs}",
    input_variables=["logs"],
)