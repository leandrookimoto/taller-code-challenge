from langchain_aws import BedrockLLM, ChatBedrock
from prompt import prompt_few_shot

class LogsAgent:

    # llm = BedrockLLM(model_id="amazon.titan-text-express-v1")
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=dict(temperature=0)
    )
    
    @classmethod
    def invoke(cls, logs:str) ->str:

        llm_chain = prompt_few_shot | cls.llm

        result = llm_chain.invoke({"logs":logs})

        return result.content

if __name__=="__main__":

    logs = """
        [2025-02-20 14:32:10] INFO - Agent Response: "Hello! How can I help you today?"
        [2025-02-20 14:33:15] ERROR - Model Timeout after 5000ms
        [2025-02-20 14:34:02] INFO - Agent Response: "I'm sorry, I didn't understand that."
        """

    result = LogsAgent.invoke(logs)
    print(result)