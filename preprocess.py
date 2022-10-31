from collections import Counter


def belongs_to_collection_func(train, test):
    train["belongs_to_collection"].apply(lambda x: len(x) if x != {} else 0).value_counts()
    train["collection_name"] = train["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else 0)
    train["has_collection"] = train["belongs_to_collection"].apply(lambda x: len(x) if x != {} else 0)

    test["collection_name"] = test["belongs_to_collection"].apply(lambda x: x[0]["name"] if x != {} else 0)
    test["has_collection"] = test["belongs_to_collection"].apply(lambda x: len(x) if x != {} else 0)

    train = train.drop(["belongs_to_collection"], axis=1)
    test = test.drop(["belongs_to_collection"], axis=1)
    return train, test


def preprocess_genders(train, test):
    train["genres"].apply(lambda x: len(x) if x != {} else 0).value_counts()
    list_of_genres = list(train["genres"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values)

    train["num_genres"] = train["genres"].apply(lambda x: len(x) if x != {} else 0)
    train["all_genres"] = train["genres"].apply(lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else "")

    top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

    for g in top_genres:
        train["genre_" + g] = train["all_genres"].apply(lambda x: 1 if g in x else 0)

    test["num_genres"] = test["genres"].apply(lambda x: len(x) if x != {} else 0)
    test["all_genres"] = test["genres"].apply(lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else "")

    for g in top_genres:
        test["genre_" + g] = test["all_genres"].apply(lambda x: 1 if g in x else 0)

    train = train.drop(["genres"], axis=1)
    test = test.drop(["genres"], axis=1)

    return train, test, list_of_genres, top_genres


def preprocess_companies(train, test):
    list_of_companies = list(
        train["production_companies"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values
    )
    train["production_companies"].apply(lambda x: len(x) if x != {} else 0).value_counts()
    train[train["production_companies"].apply(lambda x: len(x) if x != {} else 0) > 11]
    train["num_companies"] = train["production_companies"].apply(lambda x: len(x) if x != {} else 0)
    train["all_production_companies"] = train["production_companies"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
    for g in top_companies:
        train["production_company_" + g] = train["all_production_companies"].apply(lambda x: 1 if g in x else 0)

    test["num_companies"] = test["production_companies"].apply(lambda x: len(x) if x != {} else 0)
    test["all_production_companies"] = test["production_companies"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    for g in top_companies:
        test["production_company_" + g] = test["all_production_companies"].apply(lambda x: 1 if g in x else 0)

    train = train.drop(["production_companies", "all_production_companies"], axis=1)
    test = test.drop(["production_companies", "all_production_companies"], axis=1)

    return train, test, list_of_companies, top_companies


def preprocess_languages(train, test):
    train["spoken_languages"].apply(lambda x: len(x) if x != {} else 0).value_counts()

    list_of_languages = list(
        train["spoken_languages"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values
    )
    Counter([i for j in list_of_languages for i in j]).most_common(15)

    train["num_languages"] = train["spoken_languages"].apply(lambda x: len(x) if x != {} else 0)
    train["all_languages"] = train["spoken_languages"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]
    for g in top_languages:
        train["language_" + g] = train["all_languages"].apply(lambda x: 1 if g in x else 0)

    test["num_languages"] = test["spoken_languages"].apply(lambda x: len(x) if x != {} else 0)
    test["all_languages"] = test["spoken_languages"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    for g in top_languages:
        test["language_" + g] = test["all_languages"].apply(lambda x: 1 if g in x else 0)

    train = train.drop(["spoken_languages", "all_languages"], axis=1)
    test = test.drop(["spoken_languages", "all_languages"], axis=1)

    return train, test, list_of_languages, top_languages


def preprocess_keywords(train, test):
    train["Keywords"].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
    list_of_keywords = list(train["Keywords"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values)

    train["num_Keywords"] = train["Keywords"].apply(lambda x: len(x) if x != {} else 0)
    train["all_Keywords"] = train["Keywords"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
    for g in top_keywords:
        train["keyword_" + g] = train["all_Keywords"].apply(lambda x: 1 if g in x else 0)

    test["num_Keywords"] = test["Keywords"].apply(lambda x: len(x) if x != {} else 0)
    test["all_Keywords"] = test["Keywords"].apply(lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else "")
    for g in top_keywords:
        test["keyword_" + g] = test["all_Keywords"].apply(lambda x: 1 if g in x else 0)

    train = train.drop(["Keywords", "all_Keywords"], axis=1)
    test = test.drop(["Keywords", "all_Keywords"], axis=1)
    return train, test, list_of_keywords, top_keywords


def preprocess_cast(train, test):
    train["cast"].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
    list_of_cast_names = list(train["cast"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values)
    Counter([i for j in list_of_cast_names for i in j]).most_common(15)
    list_of_cast_names_url = list(
        train["cast"].apply(lambda x: [(i["name"], i["profile_path"]) for i in x] if x != {} else []).values
    )
    list_of_cast_genders = list(train["cast"].apply(lambda x: [i["gender"] for i in x] if x != {} else []).values)
    list_of_cast_characters = list(train["cast"].apply(lambda x: [i["character"] for i in x] if x != {} else []).values)

    train["num_cast"] = train["cast"].apply(lambda x: len(x) if x != {} else 0)
    top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
    for g in top_cast_names:
        train["cast_name_" + g] = train["cast"].apply(lambda x: 1 if g in str(x) else 0)
    train["genders_0_cast"] = train["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 0]))
    train["genders_1_cast"] = train["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 1]))
    train["genders_2_cast"] = train["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 2]))
    top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]
    for g in top_cast_characters:
        train["cast_character_" + g] = train["cast"].apply(lambda x: 1 if g in str(x) else 0)

    test["num_cast"] = test["cast"].apply(lambda x: len(x) if x != {} else 0)
    for g in top_cast_names:
        test["cast_name_" + g] = test["cast"].apply(lambda x: 1 if g in str(x) else 0)
    test["genders_0_cast"] = test["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 0]))
    test["genders_1_cast"] = test["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 1]))
    test["genders_2_cast"] = test["cast"].apply(lambda x: sum([1 for i in x if i["gender"] == 2]))
    for g in top_cast_characters:
        test["cast_character_" + g] = test["cast"].apply(lambda x: 1 if g in str(x) else 0)

    train = train.drop(["cast"], axis=1)
    test = test.drop(["cast"], axis=1)

    return (
        train,
        test,
        list_of_cast_names,
        list_of_cast_names_url,
        list_of_cast_genders,
        list_of_cast_characters,
        top_cast_names,
        top_cast_characters,
    )


def preprocess_crew(train, test):
    train["crew"].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
    list_of_crew_names_temp = list(train["crew"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values)
    list_of_crew_names_url = list(
        train["crew"].apply(lambda x: [(i["name"], i["profile_path"], i["job"]) for i in x] if x != {} else []).values
    )
    list_of_crew_jobs = list(train["crew"].apply(lambda x: [i["job"] for i in x] if x != {} else []).values)
    list_of_crew_genders = list(train["crew"].apply(lambda x: [i["gender"] for i in x] if x != {} else []).values)
    list_of_crew_departments = list(
        train["crew"].apply(lambda x: [i["department"] for i in x] if x != {} else []).values
    )
    list_of_crew_names = train["crew"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values

    train["num_crew"] = train["crew"].apply(lambda x: len(x) if x != {} else 0)
    top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
    for g in top_crew_names:
        train["crew_name_" + g] = train["crew"].apply(lambda x: 1 if g in str(x) else 0)
    train["genders_0_crew"] = train["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 0]))
    train["genders_1_crew"] = train["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 1]))
    train["genders_2_crew"] = train["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 2]))
    top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
    for j in top_crew_jobs:
        train["jobs_" + j] = train["crew"].apply(lambda x: sum([1 for i in x if i["job"] == j]))
    top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
    for j in top_crew_departments:
        train["departments_" + j] = train["crew"].apply(lambda x: sum([1 for i in x if i["department"] == j]))

    test["num_crew"] = test["crew"].apply(lambda x: len(x) if x != {} else 0)
    for g in top_crew_names:
        test["crew_name_" + g] = test["crew"].apply(lambda x: 1 if g in str(x) else 0)
    test["genders_0_crew"] = test["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 0]))
    test["genders_1_crew"] = test["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 1]))
    test["genders_2_crew"] = test["crew"].apply(lambda x: sum([1 for i in x if i["gender"] == 2]))
    for j in top_crew_jobs:
        test["jobs_" + j] = test["crew"].apply(lambda x: sum([1 for i in x if i["job"] == j]))
    for j in top_crew_departments:
        test["departments_" + j] = test["crew"].apply(lambda x: sum([1 for i in x if i["department"] == j]))

    train = train.drop(["crew"], axis=1)
    test = test.drop(["crew"], axis=1)
    return (
        train,
        test,
        list_of_crew_names_temp,
        list_of_crew_names,
        list_of_crew_names_url,
        list_of_crew_jobs,
        list_of_crew_genders,
        list_of_crew_departments,
        top_crew_names,
        top_crew_jobs,
    )


def preprocess_countries(train, test):
    train["production_countries"].apply(lambda x: len(x) if x != {} else 0).value_counts()

    list_of_countries = list(
        train["production_countries"].apply(lambda x: [i["name"] for i in x] if x != {} else []).values
    )
    train["num_countries"] = train["production_countries"].apply(lambda x: len(x) if x != {} else 0)
    train["all_countries"] = train["production_countries"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]
    for g in top_countries:
        train["production_country_" + g] = train["all_countries"].apply(lambda x: 1 if g in x else 0)

    test["num_countries"] = test["production_countries"].apply(lambda x: len(x) if x != {} else 0)
    test["all_countries"] = test["production_countries"].apply(
        lambda x: " ".join(sorted([i["name"] for i in x])) if x != {} else ""
    )
    for g in top_countries:
        test["production_country_" + g] = test["all_countries"].apply(lambda x: 1 if g in x else 0)

    train = train.drop(["production_countries", "all_countries"], axis=1)
    test = test.drop(["production_countries", "all_countries"], axis=1)
    return train, test, list_of_countries, top_countries


def preprocess_homepage(train, test):
    train["has_homepage"] = 0
    train.loc[train["homepage"].isnull() == False, "has_homepage"] = 1
    test["has_homepage"] = 0
    test.loc[test["homepage"].isnull() == False, "has_homepage"] = 1
    return train, test
