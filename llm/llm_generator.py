from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config.config_loader import load_config


class LLMGenerator:
    def __init__(self):
        # Load configs
        self.config = load_config("config/settings.yaml")
        self.prompts_config = load_config("config/prompts.yaml")

        # Active prompt
        self.active_prompt_name = self.config["prompt"]["active"]

        # Model
        model_name = self.config["llm"]["model_name"]
        print(f"🔹 Loading LLM: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

        # ✅ Step 4: Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,        # allow longer output
                min_new_tokens=15,         # 🔥 FORCE longer answer
                do_sample=True,            # enable sampling
                temperature=0.2,           # more expressive
                top_p=0.9,                 # better diversity
                repetition_penalty=1.2     # avoid repeating same sentence
            )

        # ✅ Step 5: Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # ------------------------------
        # CLEAN OUTPUT (POST-PROCESSING)
        # ------------------------------
        answer = answer.strip()

        # Fix duplicate / mixed "I don't know"
        if "I don't know" in answer:
            return "I don't know based on the provided documents."

        # Remove quotes / weird symbols
        answer = answer.replace('"', '').strip()

        return answer