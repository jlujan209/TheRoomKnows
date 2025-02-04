import string

print(string.punctuation)

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a string.

    Args:
        text: A string.

    Returns:
        A string with punctuation removed.
    """
    result = []
    for letter in text:
        if letter.isalpha() or letter == " ":
            result.append(letter)
    return "".join(result)

def compare(expected, actual):
    expected = remove_punctuation(expected).lower().split()
    actual = remove_punctuation(actual).lower().split()
    print(f"expected: {len(expected)}, actual: {len(actual)}")
    correct_count = 0
    incorrect_count = 0
    exp_i = 0
    act_i = 0
    for i in range(min(len(expected), len(actual))):
        if actual[act_i] == expected[exp_i]:
            correct_count += 1
            act_i += 1
            exp_i += 1
        else:
            incorrect_count += 1
            act_i += 1
            exp_i += 1
            
            print(f"{i}: expected: {expected[i]}, actual: {actual[i]}")
            while (actual[act_i] != expected[exp_i]):
                print(actual[act_i], expected[exp_i])
                exp_i += 1


with open("act_output/out.txt") as f:
    actual = f.read()
print("-----actual-----")
print(actual)
with open("exp_output/out.txt") as f:
    expected = f.read()
print("-----expected-----")
print(expected)


compare(expected, actual)

        