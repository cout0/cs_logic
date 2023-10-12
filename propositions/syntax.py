# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
           (len(string) == 1 or string[1:].isdecimal())

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'
    # For Chapter 3:
    # return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable name, or operator at the root of
            the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand for the root, if the root is a unary or
                binary operator.
            second: the second operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        # Task 1.1
        if not hasattr(self, 'first') and not hasattr(self, 'second'):
            return self.root

        if hasattr(self, 'first') and not hasattr(self, 'second'):
            return self.root + self.first.__repr__()

        return f'({self.first.__repr__()}{self.root + self.second.__repr__()})'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:
        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 1.2
        varSets = set()

        if is_variable(self.root):
            varSets.add(self.root)

        if hasattr(self, 'first'):
            varSets.update(self.first.variables())

        if hasattr(self, 'second'):
            varSets.update(self.second.variables())

        return varSets

    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        # Task 1.3
        optSets = set()

        if not is_variable(self.root):
            optSets.add(self.root)

        if hasattr(self, 'first'):
            optSets.update(self.first.operators())

        if hasattr(self, 'second'):
            optSets.update(self.second.operators())

        return optSets
        
    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union[Formula, None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator followed by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        # Task 1.4
        # null, single operators
        if len(string) == 0 or is_unary(string) or is_binary(string):
            return (None, '')

        # variable
        if is_variable(string[0]):
            maxLen = 0
            while maxLen <= len(string):
                if not is_variable(string[:maxLen + 1]):
                    break
                maxLen += 1
            return (Formula(string[:maxLen]), string[maxLen:])

        # constant
        if is_constant(string[0]):
            return (Formula(string[0]), string[1:])

        # unary
        if is_unary(string[0]):
            ff, rr = Formula._parse_prefix(string[1:])
            if ff is not None:
                return (Formula(string[0], ff), rr)
            else:
                return (None, '')

        # not (
        if string[0] != '(':
            return (None, '')

        # (
        bracCounts = 1
        rightBracIdx = 0
        while bracCounts != 0 and rightBracIdx + 1 < len(string):
            rightBracIdx += 1
            if string[rightBracIdx] == '(':
                bracCounts += 1
            if string[rightBracIdx] == ')':
                bracCounts -= 1
            if bracCounts == 0:
                break

        # () not pair
        if bracCounts != 0:
            return (None, '')

        # lchild
        lff, lrr = Formula._parse_prefix(string[1:rightBracIdx])
        if lrr == None:
            return (None, '')

        # find binary operator
        binOpt = ''
        for binOptIdx in range(3):
            if is_binary(lrr[:binOptIdx + 1]):
                binOpt = lrr[:binOptIdx + 1]
                break

        # no binary operator
        if len(binOpt) == 0:
            return (None, '')

        # rchild
        rff, rrr = Formula._parse_prefix(lrr[len(binOpt):])
        if rff is None or len(rrr) != 0:
            return (None, '')

        return (Formula(binOpt, lff, rff), string[rightBracIdx + 1:])

    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        # Task 1.5
        ff, _ = Formula._parse_prefix(string)

        return False\
            if ff is None\
            else\
            str(ff) == string
        
    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        # Task 1.6
        ff, _ = Formula._parse_prefix(string)

        return None\
            if ff is None or str(ff) != string\
            else\
            ff

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7
        string = self.root

        if hasattr(self, 'first'):
            string += self.first.polish()

        if hasattr(self, 'second'):
            string += self.second.polish()

        return string

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8
        # variable
        if is_variable(string[0]):
            maxLen = 0
            while maxLen <= len(string):
                if not is_variable(string[:maxLen + 1]):
                    break
                maxLen += 1
            return Formula(string[:maxLen])

        # constant
        if is_constant(string[0]):
            return Formula(string[0])

        # find operator
        root = ''
        for optIdx in range(3):
            if is_unary(string[:optIdx + 1]) or is_binary(string[:optIdx + 1]):
                root = string[:optIdx + 1]
                break

        # unary operator
        if is_unary(root):
            return Formula(root, Formula.parse_polish(string[len(root):]))

        # binary operator
        if is_binary(root):
            ll = Formula.parse_polish(string[len(root):])
            rr = Formula.parse_polish(string[len(root) + len(ll.polish()):])
            return Formula(root, ll, rr)

        return None

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable name `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable name occurrences originating in the current formula are
            substituted (i.e., variable name occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                   is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
