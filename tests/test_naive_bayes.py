from src.naive_bayes import NaiveBayes

def test_NaiveBayes():
    nb = NaiveBayes()
    nb.load(
        {
            "free": {"ham": 100, "spam": 300},
            "viagra": {"ham": 10, "spam": 90},
            "other": {"ham": 290, "spam": 210},
        }
    )
    prediction = nb.predict("free viagra")
    print(prediction)
    assert prediction == 0.9474098704935247, f"incorrect {prediction=}, expected: 0.9474098704935247"
