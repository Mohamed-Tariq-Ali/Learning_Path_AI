
def add_or_update(dct, key, value):
    """Add a new key or update an existing key with the given value."""
    dct[key] = value
    print("Updated Dictionary:", dct)

def remove_key(dct, key):
    """Remove a key from the dictionary if it exists."""
    if key in dct:
        del dct[key]
        print("Updated Dictionary:", dct)
    else:
        print(f"Key '{key}' not found. No changes made.")

def clear_dict(dct):
    """Clear all items in the dictionary."""
    dct.clear()
    print("Dictionary cleared:", dct)

def parse_kv_input(prompt="Enter elements (key:value,key:value): "):
    """
    Parse input of the form 'k1:v1,k2:v2' into a dictionary.
    Values are converted to int if possible; otherwise left as strings.
    """
    raw = input(prompt).strip()
    if not raw:
        return {}
    pairs = [p.strip() for p in raw.split(',') if p.strip()]
    result = {}
    for pair in pairs:
        if ':' not in pair:
            print(f"Skipping invalid pair '{pair}' (missing ':').")
            continue
        k, v = pair.split(':', 1)
        k = k.strip()
        v = v.strip()
        # Try to convert value to int; if fails, keep as string
        try:
            v = int(v)
        except ValueError:
            pass
        result[k] = v
    return result

def add_new_dict():
    """Create and return a brand-new dictionary from user input."""
    new_dct = parse_kv_input("Enter elements for NEW dict (key:value,key:value): ")
    print("New Dictionary:", new_dct)
    return new_dct

def extend_dict(dct):
    """Extend (merge) the current dictionary with user-provided key-value pairs."""
    other = parse_kv_input("Enter elements to EXTEND (key:value,key:value): ")
    dct.update(other)
    print("Updated Dictionary:", dct)

def show_sorted(dct):
    """
    Show a sorted view of the dictionary without modifying it.
    Offers sorting by 'key' or 'value'.
    """
    if not dct:
        print("Dictionary is empty. Nothing to sort.")
        return

    print("Sort by: 1) Key  2) Value")
    try:
        choice = int(input("Enter choice (1/2): ").strip())
    except ValueError:
        print("Invalid choice. Showing by key.")
        choice = 1

    if choice == 2:
        # Ensure values are comparable
        try:
            sorted_items = sorted(dct.items(), key=lambda kv: kv[1])
        except TypeError:
            print("Values are not comparable for sorting. Sorting by keys instead.")
            sorted_items = sorted(dct.items(), key=lambda kv: kv[0])
    else:
        sorted_items = sorted(dct.items(), key=lambda kv: kv[0])

    print("Sorted view (does NOT modify the dict):")
    for k, v in sorted_items:
        print(f"{k}: {v}")

def print_menu():
    print("\n1. Add/Update key"
          "\n2. Remove key"
          "\n3. Clear dictionary"
          "\n4. Create NEW dictionary (separate object)"
          "\n5. Extend/Merge into current dictionary"
          "\n6. Show sorted view"
          "\n7. Show current dictionary"
          "\n8. Exit")

def main():
    # Initialize dictionary from user
    try:
        n = int(input("Enter number of initial key-value pairs: ").strip())
    except ValueError:
        print("Invalid number, starting with empty dictionary.")
        n = 0

    my_dict = {}
    for i in range(n):
        print(f"Pair {i+1}/{n}:")
        # Accept key and value separately; auto-convert value to int if possible
        key = input("  Enter key: ").strip()
        val_raw = input("  Enter value: ").strip()
        try:
            value = int(val_raw)
        except ValueError:
            value = val_raw
        my_dict[key] = value

    print("Initial Dictionary:", my_dict)

    while True:
        print_menu()
        try:
            choice = int(input("Enter choice: ").strip())
        except ValueError:
            print("Please enter a valid number between 1 and 8.")
            continue

        match choice:
            case 1:
                key = input("Enter key to add/update: ").strip()
                val_raw = input("Enter value: ").strip()
                try:
                    value = int(val_raw)
                except ValueError:
                    value = val_raw
                add_or_update(my_dict, key, value)

            case 2:
                key = input("Enter key to remove: ").strip()
                remove_key(my_dict, key)

            case 3:
                clear_dict(my_dict)

            case 4:
                _ = add_new_dict()  # created dict returned but not merged automatically

            case 5:
                extend_dict(my_dict)

            case 6:
                show_sorted(my_dict)

            case 7:
                print("Current Dictionary:", my_dict)

            case 8:
                print("Exiting")
                break

            case _:
                print("Invalid choice. Please select 1-8.")

# if __name__ == "__main__":
#     main()
