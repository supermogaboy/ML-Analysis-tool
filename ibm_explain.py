import re
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

class WatsonTrustExplainer:
    def __init__(self, api_key, project_id):
        """
        Connects to IBM Watsonx.ai
        """
        self.creds = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
        self.project_id = project_id
        
        # We use 'ibm/granite-13b-chat-v2' because it is great at enterprise/finance tasks
        self.model = ModelInference(
            model_id="ibm/granite-13b-chat-v2",
            credentials=self.creds,
            project_id=self.project_id
        )

    def generate_explanation(self, current_regime, metrics_log):
        """
        Feeds the raw metrics into Watsonx to get a human explanation.
        """
        # 1. Construct the Prompt (The "Input" for the LLM)
        prompt = f"""
        You are a Senior Financial Risk Analyst for a major bank.
        Analyze the following model performance data and write a 2-sentence notification for a trader.
        
        CONTEXT:
        - The user is a trader deciding whether to trust the AI model today.
        - Current Market Regime: {current_regime.upper()}
        
        MODEL PERFORMANCE LOG:
        {metrics_log}
        
        INSTRUCTIONS:
        - Compare the '{current_regime}' accuracy to the Baseline.
        - If Accuracy < Baseline: Warn the user to be careful.
        - If Accuracy > Baseline: Tell the user the model is reliable.
        - Do not mention 'AUC' or technical jargon. Keep it professional and actionable.
        
        YOUR EXPLANATION:
        """
        
        # 2. Call Watsonx (The "Fine-tuning/Inference" part)
        response = self.model.generate_text(prompt=prompt, params={"max_new_tokens": 100})
        
        return response.strip()
