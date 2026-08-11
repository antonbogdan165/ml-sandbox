import numpy as np
import pandas as pd


def prepare_data(file_path):
    emails = pd.read_csv(file_path)
    emails["words"] = emails["text"].apply(lambda x: list(set(x.lower().split())))
    return emails


def spam_count(emails):
    model = {}

    for index, email in emails.iterrows():
        for word in email["words"]:
            if word not in model:
                model[word] = {"spam": 1, "ham": 1}
            else:
                if email["spam"]:
                    model[word]["spam"] += 1
                else:
                    model[word]["ham"] += 1
    return model


def predict_naive_bayes(email, model):
    total = len(emails)
    num_spam = sum(emails["spam"])
    num_ham = total - num_spam
    email = email.lower()
    words = set(email.split())
    spams = [1.0]
    hams = [1.0]

    for word in words:
        if word in model:
            spams.append(model[word]["spam"] / num_spam * total)
            hams.append(model[word]["ham"] / num_ham * total)

    prod_spams = np.prod(spams) * num_spam
    prod_hams = np.prod(hams) * num_ham

    return prod_spams / (prod_spams + prod_hams)


if __name__ == "__main__":
    emails = prepare_data("emails.csv")
    model = spam_count(emails)

    spam_test = "FREE money click here to win a cash prize now"
    prob_spam_1 = predict_naive_bayes(spam_test, model)
    print(f'Test 1 (Spam email): "{spam_test}"')
    print(f"Probability that this is spam: {prob_spam_1:.6%}")

    ham_test = "Hi team, are we still meeting today to finish our homework project?"
    prob_spam_2 = predict_naive_bayes(ham_test, model)
    print(f'\nTest 2 (Ham email): "{ham_test}"')
    print(f"Probability that this is spam: {prob_spam_2:.6%}")
