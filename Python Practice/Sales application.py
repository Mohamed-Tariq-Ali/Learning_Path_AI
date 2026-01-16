def calculate_total_sales_per_item(sales_data):
    total_sales = {}

    for transaction in sales_data:
        item = transaction["item"]
        quantity = transaction["quantity"]
        price = transaction["price"]

        amount = quantity * price

        if item in total_sales:
            total_sales[item] += amount
        else:
            total_sales[item] = amount

    return total_sales


def calculate_overall_total_sales(total_sales_per_item):
    return sum(total_sales_per_item.values())


def display_results(total_sales_per_item, overall_total):
    print("\n--- Sales Report ---")
    for item, total in total_sales_per_item.items():
        print(f"Total sales for {item}: ₹{total:.2f}")

    print(f"\nOverall total sales: ₹{overall_total:.2f}")


def main():
    sales_data = []

    print("Enter sales data (type 'done' to finish)")

    while True:
        item = input("\nEnter item name: ")

        if item.lower() == "done":
            break

        try:
            quantity = float(input("Enter quantity: "))
            price = float(input("Enter price per item: "))

            sales_data.append({
                "item": item,
                "quantity": quantity,
                "price": price
            })

        except ValueError:
            print("❌ Quantity and price must be numbers. Please try again.")

    if not sales_data:
        print("No sales data entered.")
        return

    total_sales_per_item = calculate_total_sales_per_item(sales_data)
    overall_total = calculate_overall_total_sales(total_sales_per_item)

    display_results(total_sales_per_item, overall_total)


main()
