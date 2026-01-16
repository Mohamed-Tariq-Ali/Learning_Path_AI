def count_vowels(text):
    vowels = "aeiou"
    count = 0

    for char in text.lower():
        if char in vowels:
            count += 1

    return count


def count_consonants(text):
    vowels = "aeiou"
    count = 0

    for char in text.lower():
        if char.isalpha() and char not in vowels:
            count += 1

    return count


def reverse_string(text):
    return text[::-1]


def main():
    while True:
        user_input = input("Enter a string: ")

        if user_input.strip() == "":
            print("Input cannot be empty. Please enter a valid string.\n")
            continue
        break

    vowels_count = count_vowels(user_input)
    consonants_count = count_consonants(user_input)
    reversed_text = reverse_string(user_input)

    print("\n--- String Analysis Result ---")

    if vowels_count == 0 and consonants_count == 0:
        print("No vowels or consonants found.")
    else:
        print(f"Number of vowels: {vowels_count}")
        print(f"Number of consonants: {consonants_count}")

    print(f"Reversed string: {reversed_text}")


main()
