"""Build a plan-supervision corpus for the latent planner (path B).

Each example has 4 fields:
  instruction     : a natural-language task (general, many families)
  plan            : a natural-language plan (the decomposition)
  plan_tokens     : a sequence of operation codes from the 64-op inventory + EOP
  response        : the gold worked answer

The 64 operations are a small "cognitive instruction set" (8 families x 8). Plans are
*compositions* of these ops; the corpus is 800 unique instructions with varied op-sequences.
Everything is generated deterministically (correct + reproducible, no LLM), so responses to
computable tasks (arithmetic, sort, filter, convert, count, extract, format, ...) are exact.

  python dataset/build_plan_dataset.py        # -> dataset/plan_dataset.jsonl, operations.json
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7
TARGET = 800

# ---------------------------------------------------------------- 64-op inventory
FAMILIES = {
    "GROUND": ["IDENTIFY_TASK", "PARSE_INPUT", "EXTRACT_ENTITIES", "EXTRACT_NUMBERS",
               "EXTRACT_CONSTRAINTS", "IDENTIFY_KEYWORDS", "SEGMENT_TEXT", "DETECT_LANGUAGE"],
    "RECALL": ["ENUMERATE", "RECALL_FACTS", "RETRIEVE_DEFINITION", "LIST_EXAMPLES",
               "LIST_STEPS", "LIST_PROS_CONS", "GENERATE_CANDIDATES", "BRAINSTORM"],
    "SELECT": ["SELECT_SUPERLATIVE", "FILTER_BY_CONDITION", "RANK", "TOP_K",
               "PICK_BEST", "SELECT_RELEVANT", "DEDUP", "CHOOSE_BY_CRITERIA"],
    "COMPUTE": ["COMPUTE_ARITHMETIC", "SORT", "AGGREGATE", "CONVERT_UNITS",
                "MAP_TRANSFORM", "NORMALIZE", "COUNT", "ROUND_ESTIMATE"],
    "REASON": ["COMPARE", "CLASSIFY", "EVALUATE_CONDITION", "INFER_CAUSE",
               "DECOMPOSE", "DEDUCE", "CHECK_LOGIC", "ESTIMATE"],
    "GENERATE": ["WRITE_SENTENCE", "WRITE_PARAGRAPH", "WRITE_CODE", "GIVE_EXAMPLE",
                 "CONSTRUCT_ANALOGY", "DRAFT_OUTLINE", "COMPOSE_MESSAGE", "GENERATE_QUESTION"],
    "VERIFY": ["VERIFY_FORMAT", "CHECK_CONSTRAINT", "FACT_CHECK", "CRITIQUE",
               "REVISE", "VALIDATE_NUMBER", "SELF_CORRECT", "SUMMARIZE_CHECK"],
    "COMMUNICATE": ["SUMMARIZE", "EXPLAIN_SIMPLE", "FORMAT_LIST", "FORMAT_TABLE",
                    "FORMAT_JSON", "ADAPT_TONE", "ADD_CAVEAT", "CONCLUDE"],
}
OPS = [op for fam in FAMILIES.values() for op in fam]            # 64
assert len(OPS) == 64 and len(set(OPS)) == 64
OP_ID = {op: i for i, op in enumerate(OPS)}
EOP = "EOP"
OP_ID[EOP] = 64

rng = random.Random(SEED)
rows, seen = [], set()


def humanize(tok):
    return tok.lower().replace("_", " ")


def add(instruction, tokens, response, plan=None):
    if instruction in seen:
        return
    seen.add(instruction)
    body = [t for t in tokens if t != EOP]
    if plan is None:
        plan = "Plan: " + "; ".join(f"{i+1}) {humanize(t)}" for i, t in enumerate(body)) + "."
    rows.append({
        "instruction": instruction,
        "plan": plan,
        "plan_tokens": tokens,
        "plan_token_ids": [OP_ID[t] for t in tokens],
        "response": response,
    })


# ---------------------------------------------------------------- content tables
CATEGORIES = {  # name: (members, superlative_word, answer, one-line description)
    "planets of the solar system": (["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"], "largest", "Jupiter", "a gas giant more massive than all the other planets combined"),
    "oceans of the Earth": (["Pacific", "Atlantic", "Indian", "Southern", "Arctic"], "largest", "Pacific Ocean", "it covers about a third of Earth's surface and is the deepest"),
    "continents": (["Africa", "Antarctica", "Asia", "Australia", "Europe", "North America", "South America"], "largest", "Asia", "it has the most land area and the most people"),
    "primary colors": (["red", "blue", "yellow"], "warmest", "red", "it is associated with heat and energy"),
    "states of matter": (["solid", "liquid", "gas"], "least dense", "gas", "its particles are far apart and fill any container"),
    "seasons": (["spring", "summer", "autumn", "winter"], "coldest", "winter", "days are short and temperatures drop, often with snow"),
    "five senses": (["sight", "hearing", "smell", "taste", "touch"], "most used for reading", "sight", "the eyes detect light and send signals to the brain"),
    "basic arithmetic operations": (["addition", "subtraction", "multiplication", "division"], "inverse of multiplication", "division", "it splits a number into equal parts"),
    "common sorting algorithms": (["bubble sort", "insertion sort", "merge sort", "quicksort"], "fastest on average", "quicksort", "it averages O(n log n) by partitioning around a pivot"),
    "programming languages": (["Python", "Java", "C++"], "most used for data science", "Python", "its libraries like pandas and scikit-learn make analysis easy"),
    "noble gases": (["helium", "neon", "argon", "krypton", "xenon"], "lightest", "helium", "it is the second-lightest element and is used in balloons"),
    "musical string instruments": (["violin", "viola", "cello", "double bass"], "lowest pitched", "double bass", "it is the largest and produces the deepest notes"),
    "geometric shapes": (["triangle", "square", "pentagon", "hexagon"], "with the most sides", "hexagon", "it has six sides and six angles"),
    "units of time": (["second", "minute", "hour", "day"], "longest", "day", "it equals 24 hours"),
    "food groups": (["grains", "vegetables", "fruits", "protein", "dairy"], "richest in calcium", "dairy", "milk and cheese provide calcium for bones"),
    "primary compass directions": (["north", "south", "east", "west"], "where the sun rises", "east", "the sun appears to rise in the east"),
}

CONCEPTS = {
    "photosynthesis": "the process by which plants turn sunlight, water, and carbon dioxide into sugar and oxygen",
    "gravity": "the force that pulls objects with mass toward one another",
    "inflation": "a general rise in prices that lowers the buying power of money over time",
    "evaporation": "the change of a liquid into a gas, usually due to heat",
    "a noun": "a word that names a person, place, thing, or idea",
    "an algorithm": "a step-by-step procedure for solving a problem",
    "osmosis": "the movement of water through a membrane from low to high solute concentration",
    "a prime number": "a whole number greater than 1 divisible only by 1 and itself",
    "supply and demand": "the relationship between how much of a good is available and how much people want it",
    "machine learning": "a method where computers learn patterns from data instead of being explicitly programmed",
    "the water cycle": "the continuous movement of water through evaporation, condensation, and precipitation",
    "compound interest": "interest calculated on both the original amount and the interest already earned",
}

ANALOGY = {
    "a CPU": "the brain of a computer that carries out instructions",
    "RAM": "a desk where you keep the things you are actively working on",
    "a firewall": "a security guard that checks who is allowed in and out",
    "DNA": "an instruction manual stored inside every cell",
    "a router": "a post office that directs data to the right address",
    "the immune system": "an army that defends the body against invaders",
}

SENTIMENT = [
    ("The food was delicious and the staff were so kind.", "positive"),
    ("Terrible service and the room was dirty.", "negative"),
    ("I absolutely loved this movie, what a masterpiece.", "positive"),
    ("The product broke after one day, total waste of money.", "negative"),
    ("Best purchase I have made all year.", "positive"),
    ("The flight was delayed and the seats were uncomfortable.", "negative"),
    ("A wonderful, heartwarming story with great characters.", "positive"),
    ("Overpriced and underwhelming, I would not recommend it.", "negative"),
    ("The team was helpful and solved my issue quickly.", "positive"),
    ("Worst customer support I have ever dealt with.", "negative"),
    ("Beautiful design and incredibly easy to use.", "positive"),
    ("It arrived damaged and nobody responded to my emails.", "negative"),
]

SUMMARY = [
    ("The library will be closed on Monday for maintenance and will reopen on Tuesday at 9 a.m.", "The library is closed Monday for maintenance and reopens Tuesday at 9 a.m."),
    ("Our quarterly sales rose by 12 percent, driven mainly by strong demand in the Asian market.", "Quarterly sales rose 12 percent, led by strong Asian demand."),
    ("The committee reviewed the proposal, requested two changes, and approved the revised budget.", "The committee approved the budget after requesting two changes."),
    ("Heavy rain caused flooding downtown, and several roads were closed during rush hour.", "Heavy rain flooded downtown and closed several roads at rush hour."),
    ("The new policy lets employees work from home two days a week starting next month.", "Employees may work from home two days a week starting next month."),
    ("Researchers found that the drug reduced symptoms in most patients with few side effects.", "The drug reduced symptoms in most patients with few side effects."),
    ("The festival drew record crowds this year despite the cold weather and high ticket prices.", "The festival drew record crowds despite cold weather and high prices."),
    ("After months of testing, the company recalled the device due to a battery safety risk.", "The company recalled the device over a battery safety risk."),
]

PROS_CONS = {
    "remote work": (["saves commute time", "more flexibility"], ["harder to collaborate", "can feel isolating"]),
    "electric cars": (["lower running cost", "no tailpipe emissions"], ["higher upfront price", "charging takes time"]),
    "social media": (["easy to stay connected", "fast access to news"], ["can spread misinformation", "can harm attention"]),
    "online learning": (["learn at your own pace", "no commute"], ["needs self-discipline", "less hands-on practice"]),
    "living in a city": (["more jobs and services", "good public transport"], ["higher cost of living", "more noise and crowds"]),
    "owning a pet": (["companionship", "encourages activity"], ["ongoing cost", "needs daily care"]),
}

TASK_STEPS = {
    "make a cup of tea": ["boil water", "add a tea bag to a cup", "pour in the water", "let it steep, then remove the bag"],
    "plant a seed": ["fill a pot with soil", "make a small hole", "drop in the seed and cover it", "water it and place it in light"],
    "send an email": ["open your email app", "click compose", "enter the address, subject, and message", "press send"],
    "reset a password": ["go to the login page", "click 'forgot password'", "enter your email and open the reset link", "set a new password"],
    "back up a phone": ["connect to Wi-Fi", "open settings and find backup", "choose what to back up", "start the backup and wait"],
    "tie shoelaces": ["cross the two laces", "tuck one under and pull tight", "make a loop with each lace", "cross and pull the loops into a knot"],
}

CODE_TASKS = {
    "returns the square of a number": "def square(n):\n    return n * n",
    "checks if a number is even": "def is_even(n):\n    return n % 2 == 0",
    "returns the larger of two numbers": "def maximum(a, b):\n    return a if a > b else b",
    "reverses a string": "def reverse(s):\n    return s[::-1]",
    "sums a list of numbers": "def total(xs):\n    return sum(xs)",
    "counts the vowels in a string": "def vowels(s):\n    return sum(c in 'aeiou' for c in s.lower())",
}

FLAWED = [
    ("The capital of Australia is Sydney.", "The capital of Australia is Canberra, not Sydney."),
    ("Water boils at 90 degrees Celsius at sea level.", "Water boils at 100 degrees Celsius at sea level, not 90."),
    ("A triangle has four sides.", "A triangle has three sides, not four."),
    ("The sum of 7 and 8 is 56.", "The sum of 7 and 8 is 15; 56 is their product."),
    ("Humans have three lungs.", "Humans have two lungs, not three."),
    ("The first month of the year is February.", "The first month of the year is January, not February."),
]

CLAIMS = [
    ("the Sun is a star", True), ("spiders are insects", False),
    ("water is made of hydrogen and oxygen", True), ("the Great Wall is in Egypt", False),
    ("a year has 12 months", True), ("bats are blind", False),
    ("sound travels faster than light", False), ("the heart pumps blood", True),
    ("penguins can fly", False), ("ice is frozen water", True),
]

CAUSES = {
    "a plant's leaves turn yellow": "it may not be getting enough water or nutrients",
    "a phone battery drains fast": "too many apps may be running in the background",
    "bread does not rise": "the yeast may be old or the water too hot",
    "a room feels stuffy": "there may be poor air circulation",
    "a car will not start": "the battery may be dead",
    "ice melts quickly": "the surrounding temperature is above freezing",
}

LANGS = [("bonjour le monde", "French"), ("hola mundo", "Spanish"), ("hallo welt", "German"),
         ("ciao mondo", "Italian"), ("ola mundo", "Portuguese"), ("hello world", "English")]

WORDS = ["river", "mountain", "garden", "planet", "engine", "puzzle", "guitar", "window",
         "forest", "rocket", "castle", "bridge", "candle", "pirate", "magnet", "flower",
         "harbor", "lantern", "meadow", "compass", "anchor", "feather", "marble", "tunnel",
         "orchard", "glacier", "thunder", "diamond", "saddle", "blanket", "whistle", "ladder",
         "pebble", "cactus", "harvest", "velvet", "signal", "pottery", "willow", "beacon"]
THEME_WORDS = {"animals": ["tiger", "eagle", "shark", "rabbit"], "fruits": ["mango", "apple", "grape", "cherry"],
               "colors": ["scarlet", "indigo", "amber", "violet"], "tools": ["hammer", "wrench", "drill", "pliers"]}
UNITS = [("km", "miles", 0.621371), ("kg", "pounds", 2.20462), ("meters", "feet", 3.28084),
         ("liters", "gallons", 0.264172), ("Celsius", "Fahrenheit", None), ("hours", "minutes", 60.0)]

# ---------------------------------------------------------------- templates
def gen_list_superlative():
    for name, (members, sup, ans, desc) in CATEGORIES.items():
        add(f"List the {name}, then write a sentence about the {sup} one.",
            ["ENUMERATE", "SELECT_SUPERLATIVE", "WRITE_SENTENCE", EOP],
            f"Items: {', '.join(members)}. The {sup} is {ans} — {desc}.")
        add(f"Name the {name} and rank them, then name the {sup}.",
            ["ENUMERATE", "RANK", "SELECT_SUPERLATIVE", "CONCLUDE", EOP],
            f"{', '.join(members)}. The {sup}: {ans}.")


def gen_arithmetic():
    ops = [("plus", "+", lambda a, b: a + b), ("minus", "-", lambda a, b: a - b),
           ("times", "*", lambda a, b: a * b)]
    for name, sym, fn in ops:
        for _ in range(48):
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            add(f"Compute {a} {name} {b} and show the result.",
                ["EXTRACT_NUMBERS", "COMPUTE_ARITHMETIC", "WRITE_SENTENCE", EOP],
                f"{a} {sym} {b} = {fn(a, b)}.")


def gen_validate_calc():
    for _ in range(40):
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        claimed = a + b + rng.choice([0, 0, 1, -2, 3])
        ok = claimed == a + b
        add(f"Check whether '{a} + {b} = {claimed}' is correct.",
            ["EXTRACT_NUMBERS", "COMPUTE_ARITHMETIC", "VALIDATE_NUMBER", "CONCLUDE", EOP],
            f"{a} + {b} = {a + b}, so the statement is {'correct' if ok else 'incorrect'}.")


def gen_sort_median():
    for _ in range(60):
        xs = rng.sample(range(1, 99), rng.choice([5, 7]))
        s = sorted(xs)
        med = s[len(s) // 2]
        add(f"Sort these numbers ascending and give the median: {', '.join(map(str, xs))}.",
            ["EXTRACT_NUMBERS", "SORT", "AGGREGATE", "WRITE_SENTENCE", EOP],
            f"Sorted: {', '.join(map(str, s))}. Median: {med}.")


def gen_topk_sum():
    for _ in range(48):
        xs = rng.sample(range(1, 60), 6)
        k = rng.choice([2, 3])
        top = sorted(xs, reverse=True)[:k]
        add(f"From {', '.join(map(str, xs))}, take the {k} largest and add them.",
            ["EXTRACT_NUMBERS", "SORT", "TOP_K", "AGGREGATE", "WRITE_SENTENCE", EOP],
            f"Largest {k}: {', '.join(map(str, top))}. Sum: {sum(top)}.")


def gen_filter():
    for _ in range(50):
        xs = rng.sample(range(1, 40), 7)
        cond, keep = rng.choice([("even", [x for x in xs if x % 2 == 0]),
                                 ("odd", [x for x in xs if x % 2 == 1])])
        add(f"From {', '.join(map(str, xs))}, keep only the {cond} numbers.",
            ["EXTRACT_NUMBERS", "FILTER_BY_CONDITION", "FORMAT_LIST", EOP],
            "Kept: " + (", ".join(map(str, keep)) if keep else "(none)") + ".")
    for theme, items in THEME_WORDS.items():
        pool = items + rng.sample(WORDS, 3)
        rng.shuffle(pool)
        add(f"From this list keep only the {theme}: {', '.join(pool)}.",
            ["PARSE_INPUT", "SELECT_RELEVANT", "FORMAT_LIST", EOP],
            "Kept: " + ", ".join(w for w in pool if w in items) + ".")


def gen_convert():
    for u_from, u_to, factor in UNITS:
        for _ in range(16):
            v = rng.randint(2, 80)
            if factor is None:  # Celsius -> Fahrenheit
                out = round(v * 9 / 5 + 32, 1)
            else:
                out = round(v * factor, 2)
            add(f"Convert {v} {u_from} to {u_to}.",
                ["EXTRACT_NUMBERS", "CONVERT_UNITS", "WRITE_SENTENCE", EOP],
                f"{v} {u_from} is about {out} {u_to}.")


def gen_count():
    for _ in range(60):
        w = rng.choice(WORDS)
        kind, n = rng.choice([("letters", len(w)), ("vowels", sum(c in "aeiou" for c in w))])
        add(f"How many {kind} are in the word '{w}'?",
            ["PARSE_INPUT", "COUNT", "WRITE_SENTENCE", EOP],
            f"The word '{w}' has {n} {kind}.")


def gen_transform():
    for _ in range(36):
        w = rng.choice(WORDS)
        add(f"Reverse the word '{w}'.",
            ["PARSE_INPUT", "MAP_TRANSFORM", "WRITE_SENTENCE", EOP],
            f"'{w}' reversed is '{w[::-1]}'.")
    for _ in range(34):
        w = rng.choice(WORDS)
        add(f"Convert the word '{w}' to uppercase.",
            ["PARSE_INPUT", "NORMALIZE", "WRITE_SENTENCE", EOP],
            f"'{w}' in uppercase is '{w.upper()}'.")


def gen_extract():
    texts = ["Order 3 boxes and 12 cables by May 5.", "The room is 4 meters by 6 meters.",
             "She scored 88 and 95 on the two tests.", "Bus 42 leaves at 7 and arrives at 9.",
             "We need 5 chairs and 2 tables for 30 guests.", "The recipe uses 250 grams and 3 eggs."]
    for t in texts:
        nums = re.findall(r"\d+", t)
        add(f"Extract all the numbers from: '{t}'",
            ["PARSE_INPUT", "EXTRACT_NUMBERS", "FORMAT_LIST", EOP],
            "Numbers: " + ", ".join(nums) + ".")
    caps = ["Alice met Bob in Paris on Friday.", "The Nile flows through Egypt and Sudan.",
            "Tesla and Ford compete in Detroit.", "Mount Everest sits between Nepal and China."]
    for t in caps:
        ents = [w.strip(".") for w in t.split() if w[0].isupper() and w not in ("The",)]
        add(f"Extract the proper nouns (capitalized words) from: '{t}'",
            ["PARSE_INPUT", "EXTRACT_ENTITIES", "FORMAT_LIST", EOP],
            "Proper nouns: " + ", ".join(ents) + ".")


def gen_compare_numbers():
    for _ in range(50):
        a, b = rng.sample(range(1, 999), 2)
        add(f"Which is larger, {a} or {b}?",
            ["EXTRACT_NUMBERS", "COMPARE", "CONCLUDE", EOP],
            f"{max(a, b)} is larger than {min(a, b)}.")


def gen_prime():
    def is_prime(n):
        return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))
    for _ in range(24):
        n = rng.randint(2, 60)
        add(f"Is {n} a prime number? Answer yes or no with a reason.",
            ["PARSE_INPUT", "EVALUATE_CONDITION", "CONCLUDE", EOP],
            f"{'Yes' if is_prime(n) else 'No'} — {n} is {'only divisible by 1 and itself' if is_prime(n) else 'divisible by a number other than 1 and itself'}.")


def gen_format_json():
    people = [("Alice", 30), ("Bob", 25), ("Chen", 41), ("Dara", 19), ("Evan", 52), ("Fay", 36)]
    for name, age in people:
        add(f"Format this as a JSON object with keys name and age: {name}, {age}.",
            ["PARSE_INPUT", "MAP_TRANSFORM", "FORMAT_JSON", "VERIFY_FORMAT", EOP],
            json.dumps({"name": name, "age": age}))


def gen_format_table():
    pairs = [("apple:red, banana:yellow"), ("dog:bark, cat:meow"), ("car:road, boat:water"),
             ("sun:day, moon:night")]
    for p in pairs:
        items = [x.strip().split(":") for x in p.split(",")]
        table = "| key | value |\n|---|---|\n" + "\n".join(f"| {k.strip()} | {v.strip()} |" for k, v in items)
        add(f"Turn these key:value pairs into a two-column markdown table: {p}.",
            ["PARSE_INPUT", "SEGMENT_TEXT", "FORMAT_TABLE", EOP], table)


def gen_sentiment():
    for text, label in SENTIMENT:
        add(f"Classify the sentiment (positive or negative) of: '{text}'",
            ["PARSE_INPUT", "CLASSIFY", "CONCLUDE", EOP],
            f"Sentiment: {label}.")


def gen_summary():
    for text, summ in SUMMARY:
        add(f"Summarize in one sentence: '{text}'",
            ["PARSE_INPUT", "SUMMARIZE", "SUMMARIZE_CHECK", EOP], summ)


def gen_explain():
    for c, d in CONCEPTS.items():
        add(f"Explain {c} in one simple sentence.",
            ["RETRIEVE_DEFINITION", "EXPLAIN_SIMPLE", EOP], f"{c.capitalize()} is {d}.")
    for c, a in ANALOGY.items():
        add(f"Explain {c} using a simple analogy.",
            ["RETRIEVE_DEFINITION", "CONSTRUCT_ANALOGY", EOP], f"{c.capitalize()} is like {a}.")


def gen_proscons():
    for topic, (pros, cons) in PROS_CONS.items():
        add(f"Give two pros and two cons of {topic}.",
            ["RECALL_FACTS", "LIST_PROS_CONS", "FORMAT_LIST", EOP],
            f"Pros: {pros[0]}; {pros[1]}. Cons: {cons[0]}; {cons[1]}.")


def gen_steps():
    for task, steps in TASK_STEPS.items():
        add(f"List the steps to {task}.",
            ["IDENTIFY_TASK", "DECOMPOSE", "LIST_STEPS", "FORMAT_LIST", EOP],
            " ".join(f"{i+1}) {s}." for i, s in enumerate(steps)))


def gen_code():
    for task, code in CODE_TASKS.items():
        add(f"Write a short Python function that {task}.",
            ["IDENTIFY_TASK", "DRAFT_OUTLINE", "WRITE_CODE", "VERIFY_FORMAT", EOP], code)


def gen_critique():
    for flawed, fix in FLAWED:
        add(f"Find and correct the mistake in this statement: '{flawed}'",
            ["PARSE_INPUT", "CRITIQUE", "REVISE", "SELF_CORRECT", "WRITE_SENTENCE", EOP], fix)


def gen_factcheck():
    for claim, truth in CLAIMS:
        add(f"Is it true that {claim}? Answer true or false.",
            ["PARSE_INPUT", "FACT_CHECK", "CONCLUDE", EOP],
            f"{'True' if truth else 'False'}.")


def gen_cause():
    for effect, cause in CAUSES.items():
        add(f"Give one likely reason why {effect}.",
            ["IDENTIFY_TASK", "INFER_CAUSE", "WRITE_SENTENCE", EOP],
            f"One reason: {cause}.")


def gen_language():
    for phrase, lang in LANGS:
        add(f"Identify the language of this phrase: '{phrase}'.",
            ["PARSE_INPUT", "DETECT_LANGUAGE", "CONCLUDE", EOP], f"The language is {lang}.")


def gen_keywords():
    texts = ["The hungry fox quickly crossed the frozen river at dawn.",
             "Engineers tested the new electric engine for three days.",
             "The ancient castle stood quietly above the misty valley.",
             "Scientists discovered a bright comet near the distant planet."]
    for t in texts:
        words = [w.strip(".") for w in t.split() if len(w) > 5][:3]
        add(f"Give three keywords for: '{t}'",
            ["PARSE_INPUT", "IDENTIFY_KEYWORDS", "FORMAT_LIST", EOP],
            "Keywords: " + ", ".join(words) + ".")


def gen_segment():
    texts = ["It rained. We stayed in. The day passed slowly.",
             "She woke early. The sun was bright. Birds were singing.",
             "He opened the box. Inside was a key. He smiled.",
             "The market was busy. Prices were high. We left soon."]
    for t in texts:
        parts = [s.strip() for s in t.split(".") if s.strip()]
        add(f"Split this text into separate sentences: '{t}'",
            ["PARSE_INPUT", "SEGMENT_TEXT", "FORMAT_LIST", EOP],
            " | ".join(parts))


def gen_dedup():
    for _ in range(14):
        base = rng.sample(WORDS, 4)
        lst = base + rng.sample(base, 2)
        rng.shuffle(lst)
        seen2, uniq = set(), []
        for w in lst:
            if w not in seen2:
                seen2.add(w); uniq.append(w)
        add(f"Remove duplicates from this list: {', '.join(lst)}.",
            ["PARSE_INPUT", "DEDUP", "FORMAT_LIST", EOP], "Unique: " + ", ".join(uniq) + ".")


def gen_estimate():
    items = {"the number of days in 3 years": "about 1095 days (3 x 365)",
             "the number of hours in a week": "168 hours (24 x 7)",
             "the number of minutes in a day": "1440 minutes (24 x 60)",
             "the number of seconds in an hour": "3600 seconds (60 x 60)",
             "the number of weeks in a year": "about 52 weeks",
             "the number of months in a decade": "120 months (12 x 10)"}
    for q, a in items.items():
        add(f"Estimate {q}.",
            ["IDENTIFY_TASK", "ESTIMATE", "ROUND_ESTIMATE", "WRITE_SENTENCE", EOP], f"Roughly {a}.")


def gen_deduce():
    pats = [("all cats are mammals", "Felix is a cat", "Felix is a mammal"),
            ("all squares are rectangles", "S is a square", "S is a rectangle"),
            ("all metals conduct electricity", "copper is a metal", "copper conducts electricity"),
            ("all birds have feathers", "a robin is a bird", "a robin has feathers"),
            ("all primes above 2 are odd", "p is a prime above 2", "p is odd"),
            ("all triangles have three sides", "T is a triangle", "T has three sides")]
    for premise, fact, concl in pats:
        add(f"If {premise} and {fact}, what follows?",
            ["PARSE_INPUT", "DEDUCE", "CHECK_LOGIC", "CONCLUDE", EOP], f"It follows that {concl}.")


def gen_brainstorm():
    goals = {"reducing plastic waste": ["use refillable bottles", "buy in bulk", "choose paper packaging"],
             "saving electricity at home": ["switch to LED bulbs", "unplug idle devices", "use natural light"],
             "staying focused while studying": ["silence the phone", "study in short blocks", "take notes by hand"],
             "a school science fair": ["test plant growth under colors", "build a simple circuit", "compare paper airplanes"],
             "improving team meetings": ["set an agenda", "keep them short", "end with action items"],
             "a weekend trip on a budget": ["travel off-peak", "cook some meals", "use public transport"]}
    for goal, ideas in goals.items():
        add(f"Brainstorm three ideas for {goal}.",
            ["IDENTIFY_TASK", "BRAINSTORM", "GENERATE_CANDIDATES", "FORMAT_LIST", EOP],
            " ".join(f"{i+1}) {x}." for i, x in enumerate(ideas)))


def gen_question():
    qa = [("Paris", "What is the capital of France?"), ("8", "What is 5 plus 3?"),
          ("photosynthesis", "How do plants make their food?"), ("the heart", "Which organ pumps blood?"),
          ("water", "What is H2O commonly called?"), ("Jupiter", "What is the largest planet?")]
    for ans, q in qa:
        add(f"Write a question whose answer is '{ans}'.",
            ["IDENTIFY_TASK", "GENERATE_QUESTION", EOP], q)


def gen_tone():
    pairs = [("hey, can u send the file?", "Could you please send the file when you have a moment?"),
             ("gonna be late, traffic sucks", "I will be slightly late due to heavy traffic."),
             ("this is broken, fix it now", "This appears to be broken; could it be fixed soon?"),
             ("idk what u mean", "I am not sure I understand what you mean."),
             ("thx a lot!!", "Thank you very much."),
             ("we need this asap", "We would appreciate this as soon as possible.")]
    for casual, formal in pairs:
        add(f"Rewrite this in a polite, formal tone: '{casual}'",
            ["PARSE_INPUT", "ADAPT_TONE", "WRITE_SENTENCE", EOP], formal)


def gen_recommend():
    cases = [("a laptop for a student on a tight budget", ["gaming laptop", "budget ultrabook", "premium workstation"], "budget ultrabook", "it balances price and portability for coursework"),
             ("a pet for a small apartment", ["large dog", "cat", "horse"], "cat", "it is independent and needs little space"),
             ("a drink for a hot afternoon", ["hot coffee", "iced lemonade", "soup"], "iced lemonade", "it is cold and refreshing"),
             ("a gift for a child who loves science", ["a novel", "a chemistry set", "a tie"], "a chemistry set", "it matches their interest and is hands-on"),
             ("transport for a short city trip", ["airplane", "bicycle", "cargo ship"], "bicycle", "it is cheap and good for short distances"),
             ("a plant for a dark room", ["cactus", "snake plant", "sunflower"], "snake plant", "it tolerates low light well")]
    for need, opts, best, why in cases:
        add(f"Given the need '{need}', choose the best option from: {', '.join(opts)}.",
            ["EXTRACT_CONSTRAINTS", "GENERATE_CANDIDATES", "CHOOSE_BY_CRITERIA", "PICK_BEST", "WRITE_SENTENCE", EOP],
            f"Best: {best} — {why}.")


def gen_compose_caveat():
    facts = {"the train leaves at 9 a.m.": "but check for last-minute schedule changes",
             "the medicine is taken twice daily": "but follow your doctor's specific advice",
             "the store opens at 10": "though hours may differ on holidays",
             "the recipe serves four": "adjust quantities for a different group size",
             "the warranty lasts one year": "unless the damage is accidental",
             "the road is the fastest route": "barring heavy traffic or roadwork"}
    for fact, caveat in facts.items():
        add(f"State that {fact}, and add a sensible caveat.",
            ["RECALL_FACTS", "WRITE_SENTENCE", "ADD_CAVEAT", EOP],
            f"{fact.capitalize()} — {caveat}.")
    msgs = {"a teacher for their help": "Thank you, teacher, for your patient help — it truly made a difference.",
            "a friend for a birthday gift": "Thanks so much for the thoughtful birthday gift — I love it!",
            "a colleague for covering a shift": "I really appreciate you covering my shift — thank you.",
            "a neighbor for watering the plants": "Thank you for watering the plants while I was away."}
    for who, msg in msgs.items():
        add(f"Compose a one-line thank-you message to {who}.",
            ["IDENTIFY_TASK", "COMPOSE_MESSAGE", EOP], msg)


def gen_multistep():
    for _ in range(55):
        xs = rng.sample(range(1, 40), 6)
        evens = [x for x in xs if x % 2 == 0]
        add(f"From {', '.join(map(str, xs))}, keep the even numbers and then give their sum.",
            ["EXTRACT_NUMBERS", "FILTER_BY_CONDITION", "AGGREGATE", "WRITE_SENTENCE", EOP],
            f"Even: {', '.join(map(str, evens)) or '(none)'}. Sum: {sum(evens)}.")
    for name, (members, sup, ans, desc) in list(CATEGORIES.items())[:8]:
        add(f"List the {name}, pick the {sup}, and write a short upbeat tweet about it.",
            ["ENUMERATE", "SELECT_SUPERLATIVE", "WRITE_PARAGRAPH", "ADAPT_TONE", EOP],
            f"Items: {', '.join(members)}. Pick: {ans}. Tweet: {ans} is amazing — {desc}! #funfacts")


def gen_coverage():
    examples = {"mammals": ["dog", "whale", "bat"], "fruits": ["apple", "mango", "pear"],
                "metals": ["iron", "gold", "copper"], "team sports": ["soccer", "tennis", "boxing"],
                "musical genres": ["jazz", "rock", "classical"], "shapes": ["circle", "square", "triangle"]}
    for k, v in examples.items():
        add(f"Give three examples of {k}.",
            ["IDENTIFY_TASK", "LIST_EXAMPLES", "FORMAT_LIST", EOP],
            "Examples: " + ", ".join(v) + ".")
    one = {"a prime number": "7", "an even number": "4", "a mammal": "a dog",
           "a primary color": "red", "a renewable energy source": "solar power",
           "a programming language": "Python"}
    for k, v in one.items():
        add(f"Give one concrete example of {k}.",
            ["RETRIEVE_DEFINITION", "GIVE_EXAMPLE", "WRITE_SENTENCE", EOP],
            f"For example, {v}.")
    phrases = ["the quick brown fox jumps", "a short note", "meet me at noon today please",
               "hello there", "we will arrive early tomorrow morning", "thanks"]
    for p in phrases:
        n = rng.choice([3, 4, 5])
        w = len(p.split())
        ok = w <= n
        add(f"Does the phrase '{p}' satisfy the rule 'at most {n} words'? Answer yes or no.",
            ["PARSE_INPUT", "COUNT", "CHECK_CONSTRAINT", "CONCLUDE", EOP],
            f"It has {w} words, {'within' if ok else 'over'} the limit of {n} — {'yes' if ok else 'no'}.")


# ---------------------------------------------------------------- build
for g in [gen_coverage, gen_list_superlative, gen_arithmetic, gen_validate_calc, gen_sort_median, gen_topk_sum,
          gen_filter, gen_convert, gen_count, gen_transform, gen_extract, gen_compare_numbers,
          gen_prime, gen_format_json, gen_format_table, gen_sentiment, gen_summary, gen_explain,
          gen_proscons, gen_steps, gen_code, gen_critique, gen_factcheck, gen_cause, gen_language,
          gen_keywords, gen_segment, gen_dedup, gen_estimate, gen_deduce, gen_brainstorm,
          gen_question, gen_tone, gen_recommend, gen_compose_caveat, gen_multistep]:
    g()

rng.shuffle(rows)
rows = rows[:TARGET]

# coverage check
used = set(t for r in rows for t in r["plan_tokens"] if t != EOP)
missing = [op for op in OPS if op not in used]

os.makedirs(HERE, exist_ok=True)
with open(os.path.join(HERE, "plan_dataset.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

inventory = {"n_ops": 64, "eop_id": 64,
             "families": {fam: [{"id": OP_ID[o], "name": o} for o in ops] for fam, ops in FAMILIES.items()}}
with open(os.path.join(HERE, "operations.json"), "w") as f:
    json.dump(inventory, f, indent=2)

print(f"examples: {len(rows)} (unique instructions)")
print(f"ops covered: {64 - len(missing)}/64", ("| MISSING: " + ", ".join(missing)) if missing else "(all)")
from collections import Counter
plen = Counter(len([t for t in r["plan_tokens"] if t != EOP]) for r in rows)
print("plan-length distribution:", dict(sorted(plen.items())))
print("wrote plan_dataset.jsonl + operations.json")
