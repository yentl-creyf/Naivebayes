# usage
```
from src.naive_bayes import NaiveBayes, Type

# from scratch usage
nb = NaiveBayes()
nb.mark_message(message="not spam", data_type="ham") # or Type.ham
nb.mark_message(message="spam", data_type="spam") # or Type.spam
nb.predict(message="not spam")

# you can also load your weights
# the example data from the paper this is based on
nb = NaiveBiase()

nb.load(
    {
        "free": {"ham": 100, "spam": 300},
        "viagra": {"ham": 10, "spam": 90},
        "other": {"ham": 290, "spam": 210},
    }
)
nb.predict("free viagra")

# you can also save weights
nb.save()
```
# testing
```
python -m pytest
```