import re, json
import pandas as pd


def clean_sentences(data, column):
    cleaned = []
    for sentence in data[column]:
        # insert space between sentences if one does not exist
        text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', sentence)
        # make misc replacements
        text = text.strip(). \
            replace('as well as', 'and'). \
            replace("&amp;", " and "). \
            replace('-', ' '). \
            replace('%', ' percent '). \
            replace('+', ' plus '). \
            replace("\"", "'"). \
            replace("\n", " "). \
            replace("\r", " "). \
            strip()
        cleaned.append(' '.join(text.split()))
    return cleaned


def list_to_json(data, dir):
    with open(dir, 'w') as f:
        json.dump(data, f)
    f.close()


def sample_data(n, seed, dir, col_names: list):
    data = pd.read_csv(dir)
    samples = data.sample(n=n, random_state=seed)[col_names]
    return samples


def format_json(data):
    all_lines = []
    index = data.index
    for i in index:
        this_format = {"text": data.loc[i, "Description"], "label": [], "id": i}
        all_lines.append(this_format)
    return all_lines


if __name__ == '__main__':
    n = 20
    seed = 3
    # unit_test_sentences = pd.read_csv("sentences.csv")['Text']
    alberta_samples = sample_data(n, seed, "../../Alberta full data set 2.csv", ["Description"])
    # formatted = format_json(alberta_samples)
    # print(formatted)
    # with open("alberta_cleaned_seed_" + str(seed) + "_" + str(n) + ".jl", "w") as outfile:
    #     for entry in formatted:
    #         json.dump(entry, outfile)
    #         outfile.write("\n")
    # formatted.to_json("alberta_cleaned_seed_" + str(seed) + "_" + str(n) + ".jl", orient="records", lines=True)
    print(alberta_samples)
    dic_excel = {}
    index = alberta_samples.index
    for i in index:
        dic_excel[i] = alberta_samples.loc[i, "Description"]
    with open("QA_sample" + str(seed) + "_" + str(n) + ".json", "w") as outfile:
        json.dump(dic_excel, outfile)
    outfile.close()
    # for i in index:
    #     print(i)
    #     print(alberta_samples.loc[i, "Description"])

    # alberta_cleaned = clean_sentences(alberta_samples, "Description")

    # print(unit_test_sentences)
    # print(alberta_samples["Description"])

    # alberta_samples.to_json("alberta_cleaned_seed_" + str(seed) + "_" + str(n) + ".json")
    # list_to_json(unit_test_sentences.values.tolist(), "unit_test_cleaned_seed_" + str(seed) + ".json")


