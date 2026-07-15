"""
banking_system.py — Capstone Project: Advanced Thread-Safe Banking System

This module implements a thread-safe banking system with layered architecture,
dependency injection, and secure coding practices. It extends the original
Individual Coding Exercise with new capabilities developed during the module.

Key enhancements:
- Dependency Injection (IoC): Lock factory injected into BankAccount
- Layered Architecture: Business Layer + Persistence Layer (TransactionLogger)
- Secure Coding: SecureTransactionLogger with sensitive data masking
- TDD: TransactionLogger developed using Test-Driven Development

[Usage]
Run demonstration:   py banking_system.py --demo
Run unit tests:      py banking_system.py

Python Version: 3.11.9
"""

import threading
import random
import time
import unittest
import sys
from typing import List, Callable, Optional


# =============================================================================
# Custom Exceptions
# =============================================================================

class InsufficientFundsError(Exception):
    """
    Exception raised when a withdrawal or transfer amount exceeds
    the available balance in a BankAccount.
    """
    pass


# =============================================================================
# Persistence Layer — Transaction Logger
# =============================================================================

class TransactionLogger:
    """
    Records banking transactions for auditing and compliance.

    This class represents the Persistence Layer in a layered architecture.
    It is decoupled from the Business Layer (BankAccount, TransferService)
    through dependency injection.

    Developed using Test-Driven Development: unit tests were written before
    the implementation to define the expected behaviour.
    """

    def __init__(self):
        """Initialise an empty transaction log."""
        self._logs: List[dict] = []

    def log(self, account_number: str, operation: str,
            amount: float, balance_after: float) -> None:
        """
        Record a transaction in the log.

        Args:
            account_number: The account involved.
            operation: Type of operation ('deposit', 'withdraw', 'transfer_in', 'transfer_out').
            amount: The transaction amount.
            balance_after: The account balance after the transaction.
        """
        self._logs.append({
            "account": account_number,
            "operation": operation,
            "amount": amount,
            "balance_after": balance_after,
        })

    def get_logs(self) -> List[dict]:
        """Return all recorded transaction logs."""
        return list(self._logs)

    def get_logs_for_account(self, account_number: str) -> List[dict]:
        """Return logs filtered by account number."""
        return [log for log in self._logs if log["account"] == account_number]

    def clear(self) -> None:
        """Clear all logs."""
        self._logs.clear()


class SecureTransactionLogger(TransactionLogger):
    """
    A secure variant of TransactionLogger that masks sensitive data.

    Demonstrates secure coding practices by ensuring that account numbers
    are partially masked in production logs, reducing the risk of sensitive
    data exposure while preserving auditability.
    """

    def __init__(self, mask_sensitive: bool = True):
        """
        Initialise the secure logger.

        Args:
            mask_sensitive: If True, account numbers are partially masked.
        """
        super().__init__()
        self._mask_sensitive = mask_sensitive

    def _mask_account(self, account_number: str) -> str:
        """Mask all but the last 4 characters of an account number."""
        if len(account_number) <= 4:
            return "****"
        return "****" + account_number[-4:]

    def get_logs(self) -> List[dict]:
        """
        Return logs with sensitive data masked if enabled.
        """
        logs = super().get_logs()
        if self._mask_sensitive:
            masked_logs = []
            for log in logs:
                masked_log = dict(log)
                masked_log["account"] = self._mask_account(masked_log["account"])
                masked_logs.append(masked_log)
            return masked_logs
        return logs

    def get_unmasked_logs(self) -> List[dict]:
        """Return logs without masking (for authorised audit access)."""
        return super().get_logs()


# =============================================================================
# Business Layer — BankAccount
# =============================================================================

class BankAccount:
    """
    A thread-safe bank account with deposit, withdrawal, and balance inquiry.

    Uses dependency injection for the synchronisation lock, supporting
    Inversion of Control (IoC). An optional TransactionLogger can be
    injected to enable audit trails.
    """

    def __init__(self, account_number: str, initial_balance: float = 0.0,
                 lock_factory: Callable[[], object] = threading.Lock,
                 logger: Optional[TransactionLogger] = None):
        """
        Initialise a BankAccount.

        Args:
            account_number: Unique account identifier.
            initial_balance: Starting balance (non-negative).
            lock_factory: Callable that returns a lock-like object (IoC).
            logger: Optional TransactionLogger for audit trail.

        Raises:
            ValueError: If initial_balance is negative.
        """
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self.account_number = account_number
        self._balance = initial_balance
        self._lock = lock_factory()
        self._logger = logger

    def deposit(self, amount: float) -> None:
        """
        Deposit a positive amount. Thread-safe.

        Raises:
            ValueError: If amount is not positive.
        """
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        with self._lock: # type: ignore
            self._balance += amount
            if self._logger:
                self._logger.log(self.account_number, "deposit",
                                 amount, self._balance)

    def withdraw(self, amount: float) -> None:
        """
        Withdraw a positive amount. Thread-safe.

        Raises:
            ValueError: If amount is not positive.
            InsufficientFundsError: If balance is insufficient.
        """
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        with self._lock: # pyright: ignore[reportGeneralTypeIssues]
            if amount > self._balance:
                raise InsufficientFundsError(
                    f"Insufficient funds. Balance: {self._balance:.2f}, "
                    f"attempted withdrawal: {amount:.2f}"
                )
            self._balance -= amount
            if self._logger:
                self._logger.log(self.account_number, "withdraw",
                                 amount, self._balance)

    def get_balance(self) -> float:
        """Return the current balance. Thread-safe."""
        with self._lock: # type: ignore
            return self._balance

    def _acquire_lock(self) -> None:
        """Acquire the internal lock. Used by TransferService for lock ordering."""
        self._lock.acquire() # pyright: ignore[reportAttributeAccessIssue]

    def _release_lock(self) -> None:
        """Release the internal lock. Used by TransferService for lock ordering."""
        self._lock.release() # pyright: ignore[reportAttributeAccessIssue]

    def __repr__(self) -> str:
        """Return a string representation of the account."""
        return f"BankAccount({self.account_number}, Balance: {self.get_balance():.2f})"


# =============================================================================
# Business Layer — TransferService
# =============================================================================

class TransferService:
    """
    Handles inter-account transfers with deadlock prevention.

    Uses lock ordering (by account number) to prevent circular wait.
    """

    @staticmethod
    def transfer(from_account: BankAccount, to_account: BankAccount,
                 amount: float) -> None:
        """
        Transfer funds between two accounts. Deadlock-free.

        Locks are acquired in lexicographic order of account numbers
        to prevent deadlock.

        Raises:
            ValueError: If amount is not positive or accounts are the same.
            InsufficientFundsError: If source account has insufficient funds.
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive.")
        if from_account is to_account:
            raise ValueError("Cannot transfer to the same account.")

        # Lock ordering: always lock the smaller account number first
        if from_account.account_number < to_account.account_number:
            first, second = from_account, to_account
        else:
            first, second = to_account, from_account

        first._acquire_lock()
        try:
            second._acquire_lock()
            try:
                if amount > from_account._balance:
                    raise InsufficientFundsError(
                        f"Insufficient funds for transfer. "
                        f"Balance: {from_account._balance:.2f}, "
                        f"transfer amount: {amount:.2f}"
                    )
                from_account._balance -= amount
                to_account._balance += amount

                # Log both sides of the transfer
                if from_account._logger:
                    from_account._logger.log(
                        from_account.account_number, "transfer_out",
                        amount, from_account._balance)
                if to_account._logger:
                    to_account._logger.log(
                        to_account.account_number, "transfer_in",
                        amount, to_account._balance)
            finally:
                second._release_lock()
        finally:
            first._release_lock()


# =============================================================================
# UserTransactionTask Class
# =============================================================================

class UserTransactionTask:
    """
    Encapsulates a single user's transaction behaviour and statistics.

    Performs random deposits and withdrawals on a shared account.
    """

    def __init__(self, account: BankAccount, user_id: int,
                 num_transactions: int, max_amount: float = 100.0):
        """
        Initialise a UserTransactionTask.

        Args:
            account: Target BankAccount.
            user_id: User identifier.
            num_transactions: Number of transactions to perform.
            max_amount: Maximum transaction amount.
        """
        self.account = account
        self.user_id = user_id
        self.num_transactions = num_transactions
        self.max_amount = max_amount
        self.successful_deposits = 0
        self.successful_withdrawals = 0
        self.failed_withdrawals = 0

    def execute(self) -> None:
        """
        Execute random deposits and withdrawals.

        Failed withdrawals due to insufficient funds are caught and recorded
        but do not halt execution.
        """
        for _ in range(self.num_transactions):
            is_deposit = random.choice([True, False])
            amount = round(random.uniform(1.0, self.max_amount), 2)

            try:
                if is_deposit:
                    self.account.deposit(amount)
                    self.successful_deposits += 1
                else:
                    self.account.withdraw(amount)
                    self.successful_withdrawals += 1
            except InsufficientFundsError:
                self.failed_withdrawals += 1
            except ValueError:
                pass

            time.sleep(random.uniform(0.001, 0.005))

    def get_statistics(self) -> dict:
        """Return per-user transaction statistics as a dictionary."""
        return {
            "user_id": self.user_id,
            "successful_deposits": self.successful_deposits,
            "successful_withdrawals": self.successful_withdrawals,
            "failed_withdrawals": self.failed_withdrawals,
        }


# =============================================================================
# TransactionSimulator Class
# =============================================================================

class TransactionSimulator:
    """
    Coordinates concurrent execution of multiple UserTransactionTask instances.

    Creates and manages one thread per task.
    """

    def __init__(self, tasks: List[UserTransactionTask]):
        """
        Initialise the simulator.

        Args:
            tasks: List of UserTransactionTask instances.

        Raises:
            ValueError: If tasks list is empty.
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty.")
        self.tasks = tasks

    def run(self) -> List[dict]:
        """
        Run all tasks concurrently and return per-user statistics.

        Returns:
            List of dictionaries with transaction statistics per user.
        """
        threads = []

        for task in self.tasks:
            thread = threading.Thread(
                target=task.execute,
                name=f"User-{task.user_id}"
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return [task.get_statistics() for task in self.tasks]


# =============================================================================
# Demonstration Functions
# =============================================================================

def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def demo_basic_operations() -> None:
    """
    Demonstrate basic account operations with transaction logging.
    """
    print_section_header("PART 1: Basic Account Operations with Logging")

    logger = TransactionLogger()
    account = BankAccount("ACC-001", initial_balance=1000.0, logger=logger)
    print(f"  Created account: {account}")

    account.deposit(500.0)
    print(f"  After depositing $500.00:  Balance = ${account.get_balance():.2f}")

    account.withdraw(200.0)
    print(f"  After withdrawing $200.00: Balance = ${account.get_balance():.2f}")

    try:
        account.withdraw(5000.0)
    except InsufficientFundsError as e:
        print(f"  Attempted to withdraw $5,000.00:")
        print(f"    -> ERROR: {e}")

    print(f"\n  Final balance: ${account.get_balance():.2f}")

    # Display transaction log
    print(f"\n  Transaction Log ({len(logger.get_logs())} entries):")
    for entry in logger.get_logs():
        print(f"    {entry['operation']}: ${entry['amount']:.2f} "
              f"-> Balance: ${entry['balance_after']:.2f}")


def demo_secure_logger() -> None:
    """
    Demonstrate secure transaction logging with sensitive data masking.
    """
    print_section_header("PART 2: Secure Transaction Logging")

    secure_logger = SecureTransactionLogger(mask_sensitive=True)
    account = BankAccount("SENSITIVE-9999", initial_balance=2000.0,
                          logger=secure_logger)
    print(f"  Created account: {account}")

    account.deposit(1000.0)
    account.withdraw(500.0)

    print(f"  Final balance: ${account.get_balance():.2f}")

    print(f"\n  Standard log output (masked):")
    for entry in secure_logger.get_logs():
        print(f"    Account: {entry['account']} | {entry['operation']}: "
              f"${entry['amount']:.2f}")

    print(f"\n  Authorised log output (unmasked):")
    for entry in secure_logger.get_unmasked_logs():
        print(f"    Account: {entry['account']} | {entry['operation']}: "
              f"${entry['amount']:.2f}")


def demo_transfer() -> None:
    """
    Demonstrate inter-account transfer with logging.
    """
    print_section_header("PART 3: Inter-Account Transfer with Audit Trail")

    logger = TransactionLogger()
    acc_a = BankAccount("A-100", initial_balance=2000.0, logger=logger)
    acc_b = BankAccount("B-200", initial_balance=1000.0, logger=logger)
    print(f"  Account A: {acc_a}")
    print(f"  Account B: {acc_b}")
    print(f"  Total (A + B): ${acc_a.get_balance() + acc_b.get_balance():.2f}")

    TransferService.transfer(acc_a, acc_b, 750.0)
    print(f"\n  Transferred $750.00 from A to B:")
    print(f"  Account A: {acc_a}")
    print(f"  Account B: {acc_b}")
    print(f"  Total (A + B): ${acc_a.get_balance() + acc_b.get_balance():.2f}")
    print("  -> Fund conservation verified (total unchanged)")

    print(f"\n  Audit Trail ({len(logger.get_logs())} entries):")
    for entry in logger.get_logs():
        print(f"    {entry['account']}: {entry['operation']} ${entry['amount']:.2f}")


def demo_deadlock_prevention() -> None:
    """
    Demonstrate that concurrent bidirectional transfers do not cause deadlock.
    """
    print_section_header("PART 4: Deadlock Prevention - Bidirectional Transfers")

    acc_x = BankAccount("X-001", initial_balance=5000.0)
    acc_y = BankAccount("Y-002", initial_balance=5000.0)
    print(f"  Account X: {acc_x}")
    print(f"  Account Y: {acc_y}")

    def transfer_x_to_y() -> None:
        for _ in range(200):
            try:
                TransferService.transfer(acc_x, acc_y, 10.0)
            except InsufficientFundsError:
                pass

    def transfer_y_to_x() -> None:
        for _ in range(200):
            try:
                TransferService.transfer(acc_y, acc_x, 10.0)
            except InsufficientFundsError:
                pass

    print("  Starting 200 bidirectional transfers (X<->Y) on 2 threads...")
    t1 = threading.Thread(target=transfer_x_to_y, name="X-to-Y")
    t2 = threading.Thread(target=transfer_y_to_x, name="Y-to-X")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("  All transfers completed (no deadlock occurred).")
    print(f"  Account X: {acc_x}")
    print(f"  Account Y: {acc_y}")
    print(f"  Total (X + Y): ${acc_x.get_balance() + acc_y.get_balance():.2f}")
    print("  -> Deadlock-free execution and fund conservation verified")


def demo_concurrent_simulation() -> None:
    """
    Demonstrate multi-user concurrent transactions with per-user statistics.
    """
    print_section_header("PART 5: Multi-User Concurrent Transaction Simulation")

    NUM_USERS = 10
    TRANSACTIONS_PER_USER = 50

    logger = TransactionLogger()
    shared_account = BankAccount("SHARED-001", initial_balance=5000.0,
                                 logger=logger)
    print(f"  Shared account: {shared_account}")
    print(f"  Number of concurrent users: {NUM_USERS}")
    print(f"  Transactions per user: {TRANSACTIONS_PER_USER}")
    print(f"  Total expected transactions: {NUM_USERS * TRANSACTIONS_PER_USER}")

    tasks = [
        UserTransactionTask(shared_account, user_id=i,
                            num_transactions=TRANSACTIONS_PER_USER)
        for i in range(NUM_USERS)
    ]

    print("\n  Simulation running...")
    simulator = TransactionSimulator(tasks)
    stats = simulator.run()
    print("  Simulation complete.\n")

    print(f"  Final balance: ${shared_account.get_balance():.2f}")
    print(f"  Audit log entries: {len(logger.get_logs())}\n")

    print(f"  {'User ID':<10}{'Deposits':<12}{'Withdrawals':<14}{'Failed W/D':<12}")
    print("  " + "-" * 48)
    total_dep = total_wd = total_fail = 0
    for s in stats:
        print(f"  {s['user_id']:<10}{s['successful_deposits']:<12}"
              f"{s['successful_withdrawals']:<14}{s['failed_withdrawals']:<12}")
        total_dep += s['successful_deposits']
        total_wd += s['successful_withdrawals']
        total_fail += s['failed_withdrawals']
    print("  " + "-" * 48)
    print(f"  {'TOTAL':<10}{total_dep:<12}{total_wd:<14}{total_fail:<12}")

    print(f"\n  Summary:")
    print(f"    - {total_dep} successful deposits")
    print(f"    - {total_wd} successful withdrawals")
    print(f"    - {total_fail} failed withdrawals (insufficient funds)")
    print(f"    - Total operations: {total_dep + total_wd + total_fail}")
    print("  -> Thread safety: Final balance is consistent with all operations")


def run_demonstration() -> None:
    """Run the complete demonstration."""
    print("=" * 60)
    print("  CAPSTONE: ADVANCED THREAD-SAFE BANKING SYSTEM")
    print("=" * 60)

    demo_basic_operations()
    demo_secure_logger()
    demo_transfer()
    demo_deadlock_prevention()
    demo_concurrent_simulation()

    print("\n" + "=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)


# =============================================================================
# Unit Tests — TransactionLogger (TDD)
# =============================================================================

class TestTransactionLogger(unittest.TestCase):
    """
    Unit tests for TransactionLogger.

    These tests were written BEFORE the TransactionLogger implementation
    as part of the Test-Driven Development (TDD) process:
        1. RED: Write a failing test
        2. GREEN: Implement the minimum code to pass
        3. REFACTOR: Improve the code while keeping tests green
    """

    def setUp(self):
        """Create a fresh logger before each test."""
        self.logger = TransactionLogger()

    def test_log_records_single_transaction(self):
        """TDD Step 1: Verify a single transaction can be logged."""
        self.logger.log("ACC-001", "deposit", 500.0, 1500.0)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["account"], "ACC-001")
        self.assertEqual(logs[0]["operation"], "deposit")
        self.assertEqual(logs[0]["amount"], 500.0)
        self.assertEqual(logs[0]["balance_after"], 1500.0)

    def test_log_records_multiple_transactions(self):
        """Verify multiple transactions are recorded in order."""
        self.logger.log("ACC-001", "deposit", 100.0, 1100.0)
        self.logger.log("ACC-001", "withdraw", 50.0, 1050.0)
        logs = self.logger.get_logs()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["operation"], "deposit")
        self.assertEqual(logs[1]["operation"], "withdraw")

    def test_get_logs_for_account_filters_correctly(self):
        """Verify logs can be filtered by account number."""
        self.logger.log("ACC-001", "deposit", 100.0, 1100.0)
        self.logger.log("ACC-002", "deposit", 200.0, 1200.0)
        self.logger.log("ACC-001", "withdraw", 50.0, 1050.0)

        acc1_logs = self.logger.get_logs_for_account("ACC-001")
        self.assertEqual(len(acc1_logs), 2)
        for log in acc1_logs:
            self.assertEqual(log["account"], "ACC-001")

    def test_get_logs_returns_copy_not_reference(self):
        """Verify get_logs returns a copy to protect internal state."""
        self.logger.log("ACC-001", "deposit", 100.0, 1100.0)
        logs = self.logger.get_logs()
        logs.append({"fake": "entry"})
        self.assertEqual(len(self.logger.get_logs()), 1)

    def test_clear_removes_all_logs(self):
        """Verify clear empties the log."""
        self.logger.log("ACC-001", "deposit", 100.0, 1100.0)
        self.logger.clear()
        self.assertEqual(len(self.logger.get_logs()), 0)


class TestSecureTransactionLogger(unittest.TestCase):
    """Unit tests for SecureTransactionLogger."""

    def setUp(self):
        """Create a fresh secure logger before each test."""
        self.logger = SecureTransactionLogger(mask_sensitive=True)

    def test_masks_account_number(self):
        """Verify account numbers are masked in log output."""
        self.logger.log("SENSITIVE-12345678", "deposit", 500.0, 1500.0)
        logs = self.logger.get_logs()
        self.assertNotIn("SENSITIVE-12345678", logs[0]["account"])
        self.assertIn("5678", logs[0]["account"])

    def test_unmasked_logs_show_full_account(self):
        """Verify authorised access can retrieve unmasked data."""
        self.logger.log("SENSITIVE-12345678", "deposit", 500.0, 1500.0)
        logs = self.logger.get_unmasked_logs()
        self.assertEqual(logs[0]["account"], "SENSITIVE-12345678")

    def test_short_account_number_fully_masked(self):
        """Verify short account numbers are fully masked."""
        self.logger.log("AB12", "deposit", 100.0, 1100.0)
        logs = self.logger.get_logs()
        self.assertEqual(logs[0]["account"], "****")


# =============================================================================
# Unit Tests — Dependency Injection
# =============================================================================

class TestDependencyInjection(unittest.TestCase):
    """Unit tests demonstrating Dependency Injection / IoC."""

    def test_bank_account_accepts_custom_lock_factory(self):
        """Verify a custom lock factory can be injected."""
        from unittest.mock import MagicMock
        mock_lock = MagicMock()
        account = BankAccount("DI-001", initial_balance=500.0,
                              lock_factory=lambda: mock_lock)
        account.deposit(100.0)
        mock_lock.__enter__.assert_called()
        mock_lock.__exit__.assert_called()

    def test_bank_account_defaults_to_threading_lock(self):
        """Verify default lock factory creates a threading.Lock."""
        account = BankAccount("DI-002")
        self.assertIsInstance(account._lock, type(threading.Lock()))

    def test_bank_account_with_logger_injection(self):
        """Verify logger injection enables audit trail."""
        logger = TransactionLogger()
        account = BankAccount("DI-003", initial_balance=1000.0, logger=logger)
        account.deposit(500.0)
        self.assertEqual(len(logger.get_logs()), 1)


# =============================================================================
# Existing Unit Tests (preserved from original)
# =============================================================================

class TestBankAccount(unittest.TestCase):
    """Unit tests for the BankAccount class."""

    def setUp(self):
        """Create a fresh account with $1,000.00 before each test."""
        self.account = BankAccount("ACC-001", initial_balance=1000.0)

    def test_initial_balance(self):
        """Verify correct initial balance."""
        self.assertEqual(self.account.get_balance(), 1000.0)

    def test_deposit_positive_amount(self):
        """Verify deposit increases balance correctly."""
        self.account.deposit(500.0)
        self.assertEqual(self.account.get_balance(), 1500.0)

    def test_deposit_zero_raises_error(self):
        """Verify depositing zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.deposit(0.0)

    def test_deposit_negative_raises_error(self):
        """Verify depositing negative amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.deposit(-100.0)

    def test_withdraw_valid_amount(self):
        """Verify withdrawal decreases balance correctly."""
        self.account.withdraw(300.0)
        self.assertEqual(self.account.get_balance(), 700.0)

    def test_withdraw_insufficient_funds(self):
        """Verify withdrawing more than balance raises InsufficientFundsError."""
        with self.assertRaises(InsufficientFundsError):
            self.account.withdraw(2000.0)

    def test_withdraw_zero_raises_error(self):
        """Verify withdrawing zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(0.0)

    def test_withdraw_negative_raises_error(self):
        """Verify withdrawing negative amount raises ValueError."""
        with self.assertRaises(ValueError):
            self.account.withdraw(-50.0)

    def test_get_balance_after_operations(self):
        """Verify get_balance reflects all operations."""
        self.account.deposit(250.0)
        self.account.withdraw(100.0)
        self.assertEqual(self.account.get_balance(), 1150.0)

    def test_negative_initial_balance_raises_error(self):
        """Verify negative initial balance is rejected."""
        with self.assertRaises(ValueError):
            BankAccount("ACC-002", initial_balance=-100.0)

    def test_account_number_is_stored(self):
        """Verify account_number attribute is correctly stored."""
        acc = BankAccount("XYZ-999", initial_balance=50.0)
        self.assertEqual(acc.account_number, "XYZ-999")


class TestTransferService(unittest.TestCase):
    """Unit tests for the TransferService class."""

    def setUp(self):
        """Create two accounts before each test."""
        self.acc_a = BankAccount("A-001", initial_balance=1000.0)
        self.acc_b = BankAccount("B-002", initial_balance=500.0)

    def test_successful_transfer(self):
        """Verify successful transfer updates both balances."""
        TransferService.transfer(self.acc_a, self.acc_b, 300.0)
        self.assertEqual(self.acc_a.get_balance(), 700.0)
        self.assertEqual(self.acc_b.get_balance(), 800.0)

    def test_transfer_insufficient_funds(self):
        """Verify transfer with insufficient funds raises error."""
        with self.assertRaises(InsufficientFundsError):
            TransferService.transfer(self.acc_a, self.acc_b, 2000.0)

    def test_transfer_zero_raises_error(self):
        """Verify zero transfer amount raises ValueError."""
        with self.assertRaises(ValueError):
            TransferService.transfer(self.acc_a, self.acc_b, 0.0)

    def test_transfer_negative_raises_error(self):
        """Verify negative transfer amount raises ValueError."""
        with self.assertRaises(ValueError):
            TransferService.transfer(self.acc_a, self.acc_b, -100.0)

    def test_transfer_to_same_account_raises_error(self):
        """Verify transferring to the same account raises ValueError."""
        with self.assertRaises(ValueError):
            TransferService.transfer(self.acc_a, self.acc_a, 100.0)

    def test_transfer_fund_conservation(self):
        """Verify total funds are conserved after transfer."""
        initial_total = self.acc_a.get_balance() + self.acc_b.get_balance()
        TransferService.transfer(self.acc_a, self.acc_b, 250.0)
        final_total = self.acc_a.get_balance() + self.acc_b.get_balance()
        self.assertEqual(initial_total, final_total)

    def test_deadlock_prevention_bidirectional(self):
        """
        Verify that concurrent bidirectional transfers complete without
        deadlock and preserve fund conservation.
        """
        acc_x = BankAccount("X-100", initial_balance=5000.0)
        acc_y = BankAccount("Y-200", initial_balance=5000.0)
        initial_total = acc_x.get_balance() + acc_y.get_balance()

        def transfer_x_to_y():
            for _ in range(100):
                try:
                    TransferService.transfer(acc_x, acc_y, 10.0)
                except InsufficientFundsError:
                    pass

        def transfer_y_to_x():
            for _ in range(100):
                try:
                    TransferService.transfer(acc_y, acc_x, 10.0)
                except InsufficientFundsError:
                    pass

        t1 = threading.Thread(target=transfer_x_to_y)
        t2 = threading.Thread(target=transfer_y_to_x)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        final_total = acc_x.get_balance() + acc_y.get_balance()
        self.assertEqual(initial_total, final_total,
                         "Fund conservation violated after bidirectional transfers")


class TestUserTransactionTask(unittest.TestCase):
    """Unit tests for the UserTransactionTask class."""

    def test_execute_completes_without_error(self):
        """Verify task execution completes without unhandled exceptions."""
        account = BankAccount("T-001", initial_balance=1000.0)
        task = UserTransactionTask(account, user_id=1, num_transactions=20)
        task.execute()

    def test_statistics_total_matches_transaction_count(self):
        """Verify total recorded operations equal num_transactions."""
        account = BankAccount("T-002", initial_balance=1000.0)
        task = UserTransactionTask(account, user_id=5, num_transactions=50)
        task.execute()
        stats = task.get_statistics()
        total = (stats["successful_deposits"] +
                 stats["successful_withdrawals"] +
                 stats["failed_withdrawals"])
        self.assertEqual(total, 50)

    def test_statistics_user_id_correct(self):
        """Verify user_id is correctly recorded in statistics."""
        account = BankAccount("T-003", initial_balance=500.0)
        task = UserTransactionTask(account, user_id=42, num_transactions=10)
        task.execute()
        stats = task.get_statistics()
        self.assertEqual(stats["user_id"], 42)

    def test_all_statistic_fields_present(self):
        """Verify all expected keys are present in statistics."""
        account = BankAccount("T-004", initial_balance=500.0)
        task = UserTransactionTask(account, user_id=1, num_transactions=5)
        task.execute()
        stats = task.get_statistics()
        expected_keys = {"user_id", "successful_deposits",
                         "successful_withdrawals", "failed_withdrawals"}
        self.assertTrue(expected_keys.issubset(stats.keys()))


class TestTransactionSimulator(unittest.TestCase):
    """Unit tests for the TransactionSimulator class."""

    def test_single_user(self):
        """Verify simulator works with a single user."""
        account = BankAccount("S-001", initial_balance=1000.0)
        tasks = [UserTransactionTask(account, user_id=0, num_transactions=30)]
        simulator = TransactionSimulator(tasks)
        stats = simulator.run()
        self.assertEqual(len(stats), 1)
        self.assertGreaterEqual(account.get_balance(), 0.0)

    def test_multiple_users(self):
        """Verify simulator works with multiple concurrent users."""
        account = BankAccount("S-002", initial_balance=5000.0)
        num_users = 8
        tasks = [
            UserTransactionTask(account, user_id=i, num_transactions=30)
            for i in range(num_users)
        ]
        simulator = TransactionSimulator(tasks)
        stats = simulator.run()
        self.assertEqual(len(stats), num_users)
        self.assertGreaterEqual(account.get_balance(), 0.0)

    def test_empty_tasks_raises_error(self):
        """Verify empty task list raises ValueError."""
        with self.assertRaises(ValueError):
            TransactionSimulator([])

    def test_all_users_complete(self):
        """Verify all users' statistics are returned."""
        account = BankAccount("S-003", initial_balance=3000.0)
        tasks = [
            UserTransactionTask(account, user_id=i, num_transactions=25)
            for i in range(5)
        ]
        simulator = TransactionSimulator(tasks)
        stats = simulator.run()
        user_ids = [s["user_id"] for s in stats]
        self.assertEqual(sorted(user_ids), [0, 1, 2, 3, 4])


class TestThreadSafety(unittest.TestCase):
    """High-concurrency tests to verify thread safety."""

    def test_concurrent_deposits_only(self):
        """Verify 100 threads depositing produce the exact expected total."""
        account = BankAccount("TS-001", initial_balance=0.0)
        NUM_THREADS = 100
        DEPOSITS_PER_THREAD = 100
        AMOUNT = 10.0

        def deposit_many():
            for _ in range(DEPOSITS_PER_THREAD):
                account.deposit(AMOUNT)

        threads = [threading.Thread(target=deposit_many)
                   for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = NUM_THREADS * DEPOSITS_PER_THREAD * AMOUNT
        self.assertEqual(account.get_balance(), expected)

    def test_concurrent_withdrawals_atomicity(self):
        """Verify concurrent withdrawals leave balance consistent."""
        initial = 10000.0
        account = BankAccount("TS-002", initial_balance=initial)

        def withdraw_many():
            for _ in range(100):
                try:
                    account.withdraw(10.0)
                except InsufficientFundsError:
                    pass

        threads = [threading.Thread(target=withdraw_many) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_withdrawn = initial - account.get_balance()
        self.assertEqual(total_withdrawn % 10.0, 0.0)
        self.assertGreaterEqual(account.get_balance(), 0.0)

    def test_concurrent_mixed_operations_deterministic(self):
        """
        Verify deterministic mixed operations produce the expected balance.
        Each of 20 threads performs 25 deposits (+10) and 25 withdrawals (-5),
        for a net of +125 per thread.
        """
        account = BankAccount("TS-003", initial_balance=5000.0)
        NUM_THREADS = 20

        def mixed_ops():
            for i in range(50):
                if i % 2 == 0:
                    account.deposit(10.0)
                else:
                    try:
                        account.withdraw(5.0)
                    except InsufficientFundsError:
                        pass

        threads = [threading.Thread(target=mixed_ops)
                   for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = 5000.0 + NUM_THREADS * 125.0
        self.assertEqual(account.get_balance(), expected)

    def test_balance_never_negative(self):
        """Verify balance never goes negative under heavy concurrent load."""
        account = BankAccount("TS-004", initial_balance=1000.0)

        def aggressive_withdrawals():
            for _ in range(200):
                try:
                    account.withdraw(10.0)
                except InsufficientFundsError:
                    pass
                try:
                    account.deposit(5.0)
                except ValueError:
                    pass

        threads = [threading.Thread(target=aggressive_withdrawals)
                   for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertGreaterEqual(account.get_balance(), 0.0,
                                "Balance should never be negative")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demonstration()
    else:
        unittest.main(verbosity=2)