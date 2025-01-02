from collections import defaultdict
from typing import List, Dict

templates = defaultdict(dict)

templates["origin"]["example"] = "Input:{}\nOutput:{}\n"
templates["origin"]["query"] = "Input:{}\nOutput:"

templates["origin_with_space"]["example"] = "Input: {}\nOutput: {}\n"
templates["origin_with_space"]["query"] = "Input: {}\nOutput: "

templates["origin_with_newline"]["example"] = "Input:\n{}\nOutput:\n{}\n"
templates["origin_with_newline"]["query"] = "Input:\n{}\nOutput:\n"

templates["sentiment"]["example"] = "Sentence: {}\nSentiment: {}\n"
templates["sentiment"]["query"] = "Sentence: {}\nSentiment: "
templates["sentiment"]["bias_example"] = "Sentence:\nSentiment: {}\n"
templates["sentiment"]["bias_query"] = "Sentence:\nSentiment: "

templates["sentiment_channel"]["example"] = "Sentiment: {}\nSentence: {}\n"
# templates["sentiment_channel"]["query"] = "Sentiment: {}\nSentence: "
templates["sentiment_channel"]["bias_example"] = "Sentiment:\nSentence: {}\n"
# templates["sentiment_channel"]["bias_query"] = "Sentiment:\nSentence: "

templates["NLI"]["example"] = "Premise: {}\nHypothesis: {}\nRelationship: {}\n"
templates["NLI"]["query"] = "Premise: {}\nHypothesis: {}\nRelationship: "
templates["NLI"]["bias_example"] = "Premise:\nHypothesis:\nRelationship: {}\n"
templates["NLI"]["bias_query"] = "Premise:\nHypothesis:\nRelationship: "

templates["NLI_channel"]["example"] = "Relationship: {}\nPremise: {}\nHypothesis: {}\n"
# templates["NLI_channel"]["query"] = "Relationship: {}\nPremise: \nHypothesis: "
templates["NLI_channel"]["bias_example"] = "Relationship:\nPremise: {}\nHypothesis: {}\n"
# templates["NLI_channel"]["bias_query"] = "Relationship:\nPremise: \nHypothesis: "


class PromptConstructor:
    def __init__(self):
        self.template: Dict[str, str] = {}
        self.examples: List[Dict[str, str]] = []
        self.prefix: str = ""
        self.infix: str = ""
        self._cached_prompt_before_query: str | None = None
        self._prompt_before_query_needs_reconstruct: bool = True
        self._is_channel: bool = False

    def set_template(self, template_name: str) -> None:
        if "channel" in template_name:
            self._is_channel = True
        self.template = templates[template_name]
        self._prompt_before_query_needs_reconstruct = True

    def set_examples(self, examples: List[Dict[str, str]]) -> None:
        self.examples = examples
        self._prompt_before_query_needs_reconstruct = True

    def set_prefix(self, prefix: str) -> None:
        self.prefix = prefix
        self._prompt_before_query_needs_reconstruct = True

    def set_infix(self, infix: str) -> None:
        self.infix = infix
        self._prompt_before_query_needs_reconstruct = True

    @property
    def prompt_before_query(self) -> str:
        if self._prompt_before_query_needs_reconstruct:
            prompt = self.prefix
            for example in self.examples:
                if self._is_channel:
                    prompt += self.template["example"].format(example["output"], example["input"])
                else:
                    prompt += self.template["example"].format(example["input"], example["output"])
            prompt += self.infix
            self._cached_prompt_before_query = prompt
            self._prompt_before_query_needs_reconstruct = False
        return self._cached_prompt_before_query

    def get_prompt(self, query: Dict[str, str], same_format_as_example: bool = False) -> str:
        prompt = self.prompt_before_query
        if same_format_as_example:
            if self._is_channel:
                prompt += self.template["example"].format(query["output"], query["input"]).strip()
            else:
                prompt += self.template["example"].format(query["input"], query["output"]).strip()
        else:
            if self._is_channel:
                pass
            else:
                prompt += self.template["query"].format(query["input"])
        return prompt

    def get_null_input_prompt(self, query: Dict[str, str], same_format_as_example: bool = False) -> str:
        prompt = self.prompt_before_query
        if same_format_as_example:
            if self._is_channel:
                prompt += self.template["bias_example"].format(query["input"]).strip()
            else:
                prompt += self.template["bias_example"].format(query["output"]).strip()
        else:
            if self._is_channel:
                pass
            else:
                prompt += self.template["bias_query"]
        return prompt

    @staticmethod
    def get_template_names() -> List[str]:
        return list(templates.keys())


def test_unit():
    pc = PromptConstructor()

    pc.set_template("origin")
    pc.set_prefix("The following are some examples of arithmetic expressions:\n")
    pc.set_examples([{"input": "1+2", "output": "3"}, {"input": "3-2", "output": "1"}])
    pc.set_infix("What is the result of the following arithmetic expression?\n")

    print(pc.get_prompt({"input": "2*2"}))

    print()

    print(pc.get_prompt({"input": "3/2", "output": "1.5"}, same_format_as_example=True))
