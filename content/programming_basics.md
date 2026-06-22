# Introduction to Programming — Core Concepts Reference

## Variables and Data Types
A variable is a named storage location in memory that holds a value which can
change during program execution. Common data types include integers (whole
numbers), floats (decimal numbers), strings (text), and booleans (True/False).

Example:
    age = 21
    name = "Sam"
    is_student = True

A common beginner mistake is confusing assignment (=) with equality comparison
(==). Assignment stores a value; equality comparison checks whether two values
are the same and returns a boolean.

## Conditionals (if / elif / else)
Conditionals let a program make decisions by executing different code blocks
depending on whether a condition is True or False.

Example:
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    else:
        grade = "C"

Common debugging issue: forgetting that elif/else only run if the prior
condition(s) were False. Students often write multiple independent `if`
statements when they actually want `elif`, causing unintended multiple
matches.

## Loops (for / while)
A `for` loop iterates over a sequence (like a list or range) a known or
bounded number of times. A `while` loop repeats as long as a condition stays
True, which is useful when the number of iterations isn't known in advance.

Example:
    for i in range(5):
        print(i)

    count = 0
    while count < 5:
        print(count)
        count += 1

Common debugging issue: infinite loops happen when the loop's controlling
variable is never updated inside the loop body (e.g. forgetting `count += 1`).
Off-by-one errors are also common, especially with `range()`, which excludes
its endpoint.

## Functions
A function is a reusable block of code that performs a specific task. It can
accept inputs (parameters) and return an output.

Example:
    def add(a, b):
        return a + b

Common debugging issue: confusing `print()` (displays a value) with `return`
(sends a value back to the caller so it can be used elsewhere). A function
without a `return` statement implicitly returns `None`.

## Debugging Fundamentals
Debugging is the process of locating and fixing errors in code. A practical
approach:
1. Read the error message carefully — it usually names the line and type of
   error (SyntaxError, TypeError, IndexError, etc).
2. Isolate the problem by testing smaller pieces of the code.
3. Use print statements or a debugger to inspect variable values at different
   points in execution.
4. Check assumptions — confirm that variables hold the values you think they
   hold, especially after loops or conditionals.

Common error types:
- SyntaxError: the code violates the language's grammar (e.g. missing colon,
  unmatched parentheses).
- TypeError: an operation is applied to an incompatible type (e.g. adding a
  string and an integer directly).
- IndexError: trying to access a list position that doesn't exist.
- NameError: using a variable that hasn't been defined yet.

## Recursion
Recursion is when a function calls itself to solve a smaller instance of the
same problem. Every recursive function needs a base case (a condition where it
stops calling itself) to avoid infinite recursion.

Example:
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)

Common debugging issue: missing or incorrect base case, causing infinite
recursion and a stack overflow / maximum recursion depth error.

## Algorithmic Thinking
Algorithmic thinking is the process of breaking a real-world problem into a
sequence of clear, logical steps before writing any code. This includes
identifying inputs, desired outputs, and the steps needed to transform one
into the other. Students often struggle here not because of syntax, but
because they try to write code before they've planned the logic.
