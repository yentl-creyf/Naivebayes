from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)
# based on http://cs.wellesley.edu/~anderson/writing/naive-bayes.pdf


class Type(str, Enum):
    ham = "ham"
    spam = "spam"


@dataclass
class Token:
    ham: int = 1
    spam: int = 1

    def add(self, amount: int, data_type: Type):
        x = getattr(self, data_type)
        setattr(self, data_type, x + amount)

    def total(self):
        return self.ham + self.spam


class NaiveBayes:
    def __init__(self):
        self.tokens = dict()
        self.all_messages = Token()

    def __message_to_tokens(self, message: str) -> str:
        return message.lower().replace(".", "").replace(",", "").split(" ")

    def __count(self, token: str = None) -> Token:
        if token:
            return self.tokens.get(token)

        # recalc all_messages
        t = Token()
        for k, v in self.tokens.items():
            v: Token
            t.add(v.ham, "ham")
            t.add(v.spam, "spam")
        self.all_messages = t
        return self.all_messages

    def mark_message(self, message: str, data_type: Type) -> None:
        tokens = self.__message_to_tokens(message)

        for token in tokens:
            _token = self.tokens.get(token, Token())
            _token.add(1, data_type)
            self.tokens[token] = _token
            self.all_messages.add(1, data_type)
        return

    def predict(self, message: str) -> float:
        tokens = self.__message_to_tokens(message)

        # p = probability
        total = self.all_messages.total()
        p_ham = self.all_messages.ham / total
        p_spam = self.all_messages.spam / total

        logger.debug(f"p_ham={p_ham:.3f}, p_spam={p_spam:.3f}")

        # calculate the probability for each token (word)
        for token in tokens:
            _token = self.tokens.get(token, Token())
            p_ham *= _token.ham / self.all_messages.ham
            p_spam *= _token.spam / self.all_messages.spam
            logger.debug(f"token={token}, p_ham={p_ham:.3f}, p_spam={p_spam:.3f}")
        # return the overall probabilty
        return p_spam / (p_spam + p_ham)

    def load(self, tokens: dict) -> dict:
        for k, v in tokens.items():
            _token = Token(**v)
            self.tokens[k] = _token
            self.all_messages.add(_token.ham, "ham")
            self.all_messages.add(_token.spam, "spam")
        return self.tokens

    def save(self) -> dict:
        return {k: asdict(v) for k, v in self.tokens.items()}
