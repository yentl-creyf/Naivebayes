from naive_bayes import NaiveBayes, Type
import logging
import sys

# setup logging
stream_handler = logging.StreamHandler(sys.stdout)
# # log formatting
formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s")
# apply formatter to handler
stream_handler.setFormatter(formatter)
# list of handlers in case you want to have multiple e.g. file handler?
handlers = [stream_handler]
# apply handlers
logging.basicConfig(level=logging.DEBUG, handlers=handlers)

# NaiveBayes Function
nb = NaiveBayes()

# example messages
messages = [
    {"message": "first to give me 1m wont regret it", "spam": 1},
    {"message": "BET High low 50k gp minimum", "spam": 1},
    {"message": "bet now! 100k gp minimum", "spam": 1},
    {"message": "howmuch gp do you have?", "spam": 0},
]

# mark messages
for m in messages:
    nb.mark_message(message=m["message"], data_type="spam" if m["spam"] else "ham")

print(nb.predict("i have 100k gp"))
