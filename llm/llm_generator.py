from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config.config_loader import load_config
import os



class LLMGenerator:
    def __init__(self):
        # Load configs
        self.config = load_config("config/settings.yaml")
        self.prompts_config = load_config("config/prompts.yaml")

        # Active prompt
        self.active_prompt_name = self.config["prompt"]["active"]

        # Model
        model_name = os.getenv("LLM_MODEL", self.config["llm"]["model_name"])
        print(f"🔹 Loading LLM: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32
        )

    def format_context(self, documents):
        """
        Clean context without noisy chunk labels
        """

        if isinstance(documents, str):
            return documents

        return "\n\n".join(documents)

    def build_prompt(self, context, question):
        """
        Builds prompt dynamically from YAML config
        """

        try:
            prompt_template = self.prompts_config["prompts"][self.active_prompt_name]["template"]
        except KeyError:
            raise ValueError(f"❌ Prompt '{self.active_prompt_name}' not found in prompts.yaml")

        prompt = prompt_template.format(
            context=context,
            question=question
        )

        return prompt

    def generate(self, documents, question, max_new_tokens=256):
        """
        Generate answer using selected prompt
        """

        # ✅ Step 1: Format context properly
        formatted_context = self.format_context(documents)

        # ✅ Step 2: Build prompt
        prompt = self.build_prompt(formatted_context, question)

        # ✅ Step 3: Tokenize with stricter control
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # IMPORTANT for FLAN-T5
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # ✅ Step 4: Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,            # ✅ deterministic
                temperature=0.0,            # ✅ no randomness
                repetition_penalty=1.2,
                early_stopping=True,         # ✅ stop when done
                use_cache=True
            )

        # ✅ Step 5: Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # ------------------------------
        # CLEAN OUTPUT (POST-PROCESSING)
        # ------------------------------
        answer = answer.strip()

        # Fix duplicate / mixed "I don't know"
        if "not available" in answer or "i don't know" in answer:
            return "Not available in the provided context."

        # Remove quotes / weird symbols
        answer = answer.replace('"', '').strip()

        return answer