from langchain_experimental.chat_models.llm_wrapper import ChatWrapper


class MistralInstruct(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "mistral-instruct"

    sys_beg: str = "<s>[INST] "
    sys_end: str = "\n\n"
    ai_n_beg: str = "\n"
    ai_n_end: str = "</s>"
    usr_n_beg: str = "[INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"


class MistralChat(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "mistral-chat"
    
    sys_beg: str = "<|im_start|>system\n"
    sys_end: str = "<|im_end|>\n<|im_start|>user\n"
    ai_n_beg: str = "<|im_start|>assistant\n"
    ai_n_end: str = "<|im_end|>"
    usr_n_beg: str = "<|im_start|>user\n"
    usr_n_end: str = "<|im_end|>\n"
    usr_0_beg: str = ""
    usr_0_end: str = "<|im_end|>\n"


class ZephyrChat(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "zephyr-chat"
    
    sys_beg: str = "<|system|>\n"
    sys_end: str = "</s>\n<|user|>\n"
    ai_n_beg: str = "<|assistant|>\n"
    ai_n_end: str = "</s>"
    usr_n_beg: str = "<|user|>\n"
    usr_n_end: str = "</s>\n"
    usr_0_beg: str = ""
    usr_0_end: str = "</s>\n"
    