class BankAccount:
    def __init__(self):
        self.__balance = 0.0   # PRIVATE variable

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")

        self.__balance += amount
        print(f"Deposited: ₹{amount:.2f}. New Balance: ₹{self.__balance:.2f}")

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")

        if amount > self.__balance:
            raise ValueError(
                f"Insufficient funds. Available balance: ₹{self.__balance:.2f}"
            )

        self.__balance -= amount
        print(f"Withdrew: ₹{amount:.2f}. New Balance: ₹{self.__balance:.2f}")

    def check_balance(self):
        print(f"Current Balance: ₹{self.__balance:.2f}")
        return self.__balance


def get_valid_amount(message):
    """Keeps asking user until valid numeric input is entered"""
    while True:
        try:
            amount = float(input(message))
            return amount
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def main():
    account = BankAccount()

    while True:
        print("\n--- Simple Banking Application ---")
        print("1. Deposit")
        print("2. Withdraw")
        print("3. Check Balance")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        try:
            if choice == "1":
                amount = get_valid_amount("Enter amount to deposit: ")
                account.deposit(amount)

            elif choice == "2":
                amount = get_valid_amount("Enter amount to withdraw: ")
                account.withdraw(amount)

            elif choice == "3":
                account.check_balance()

            elif choice == "4":
                print("Thank you for using the Banking Application.")
                break

            else:
                print("Invalid choice. Please select between 1 and 4.")

        except ValueError as e:
            print("Error:", e)


main()
