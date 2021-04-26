from lxml import html
import numpy as np
import requests
import math


default_coefficients = [-7.0767435965352634,
                        -0.78070084607335222,
                        4.8652217212091875,
                        2.4968433676873092,
                        3.4933404610013374,
                        4.9306457933253691,
                        6.1823241977326413,
                        4.6266561430466036]


def pron_feature(file_name, headline):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\t\n\r') for x in content]
    listed_headline = headline.split(" ")
    listed_headline = [x.strip('\'') for x in listed_headline]
    feature_value = 0

    for word in listed_headline:
        for pattern in content:
            if len(pattern) == 1:
                new_word = word.split("'")
                if new_word[0] == pattern:
                    feature_value += 1

            elif word.find(pattern) == 0:
                new_word = word.split("'")
                if new_word[0] == pattern:
                    feature_value += 1

        for pattern in content:
            if pattern == pattern.title():
                continue
            pattern = pattern.title()
            new_word = word.split("'")
            if new_word[0] == pattern:
                feature_value += 1

        for pattern in content:
            if pattern == pattern.upper():
                continue
            pattern = pattern.upper()
            if new_word[0] == pattern:
                    feature_value += 1

    return feature_value


def sym_feature(file_name, headline):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\t\n\r') for x in content]

    listed_headline = headline.split(" ")
    listed_headline = [x.strip('\'') for x in listed_headline]
    feature_value = 0

    for word in listed_headline:
        for pattern in content:

            if word.find(pattern) >= 0:
                feature_value += 1

    return feature_value


def words_feature(file_name, headline):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\t\n\r') for x in content]

    listed_headline = headline.split(" ")
    listed_headline = [x.strip('\'') for x in listed_headline]
    feature_value = 0

    for word in listed_headline:
        for pattern in content:
            if word.find(pattern) == 0:
                feature_value += 1

        for pattern in content:
            pattern = pattern.title()
            if word.find(pattern) == 0:
                feature_value += 1

        for pattern in content:
            pattern = pattern.upper()
            if word.find(pattern) == 0:
                feature_value += 1

    return feature_value


def headline_to_vector(headline):
    features_vector = np.zeros(7, dtype=np.int)
    feature_index = 0

    features_vector[feature_index] = words_feature("clickbaitdetector\Feelings.txt", headline)
    # print("Felling Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = pron_feature("clickbaitdetector\DPronouns.txt", headline)
    # print("Pronouns Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = sym_feature("clickbaitdetector\KeySymbols.txt", headline)
    # print("KeySymbols Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = words_feature("clickbaitdetector\KeyAdj.txt", headline)
    # print("KeyNouns Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = words_feature("clickbaitdetector\KeyNouns.txt", headline)
    # print("KeyWords Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = words_feature("clickbaitdetector\KeyVerbs.txt", headline)
    # print("KeyWords Found: ", features_vector[feature_index])
    feature_index += 1

    features_vector[feature_index] = words_feature("clickbaitdetector\IndPronouns.txt", headline)
    # print("KeyWords Found: ", features_vector[feature_index])

    return features_vector


def download_BBC_headlines():
    page = requests.get('http://www.bbc.com/')
    tree = html.fromstring(page.content)
    headlines = tree.xpath('//a[@class="media__link"] /text()')
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    return headlines


def download_WSJ_headlines():
    page = requests.get('https://www.wsj.com/europe')
    tree = html.fromstring(page.content)
    headlines = tree.xpath('//a[@class="wsj-headline-link"] /text()')
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    headlines = [x for x in headlines if x != '']
    return headlines


def download_NYT_headlines():
    page = requests.get('https://www.nytimes.com/')
    tree = html.fromstring(page.content)
    headlines = tree.xpath('//h2[@class="story-heading"]/a /text()')
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    headlines = [x for x in headlines if x != '']
    return headlines


def download_Mediaite_headlines():
    page = requests.get('http://www.mediaite.com/')
    tree = html.fromstring(page.content)
    headlines = tree.xpath('//div[@class="post post-chron"]//a /text()')
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    headlines = [x for x in headlines if x != '']
    return headlines


def download_headlines(url, xpath):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    headlines = tree.xpath(xpath)
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    headlines = [x for x in headlines if x != '']
    return headlines


def __save_to_file(file_name, headlines):
    f = open(file_name, "ab")
    for headline in headlines:
        new_s = headline.encode('UTF-8')
        f.write(new_s)
        f.write(bytes('\n\r', 'UTF-8'))


def parse_training_data(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\t\n\r') for x in content]
    content = [x for x in content if x != '']
    dataset = []
    for line in content:
        line = line.split('^')
        # print(line)
        if len(line) == 3:
            headline = line[0]
            features_vector = np.fromstring(line[1].strip('\[\]'),
                                            dtype=np.int, sep=' ')
            # features_vector = line[1]
            target = float(line[2])
            print(headline, features_vector, target)
            if target >= 0.5:
                target = 1
            elif target < 0.5:
                target = 0
            training_vector = np.zeros(8, dtype=np.int)
            training_vector[:len(features_vector)] = features_vector
            training_vector[7] = target
            dataset.append(training_vector)
    return dataset


# Make a prediction with coefficients
def __predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + math.exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = __predict(row, coef)
            error = row[-1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] += l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


def check_clickbait(headline, coefficients):
    features_vector = headline_to_vector(headline)
    result = 0
    for i in range(1, 8):
        result += features_vector[i - 1] * coefficients[i]
    result += coefficients[0]
    result = 1 / (1 + math.exp(-result))
    return result


def check_clickbait_url(url, xpath, coefficients):
    page = requests.get(url)
    tree = html.fromstring(page.content)
    headlines = tree.xpath(xpath)
    headlines = [headline.strip(' \t\n\r') for headline in headlines]
    headlines = [x for x in headlines if x != '']
    result = []
    print(len(headlines))
    for headline in headlines:
        t = (check_clickbait(headline, coefficients), str(headline))
        result.append(t)
    return result


def check_clickbait_file(file_name, coefficients):
    headlines = load_headlines_file(file_name)
    result = []
    print(len(headlines))
    for headline in headlines:
        t = (check_clickbait(headline, coefficients), str(headline))
        result.append(t)
    return result
