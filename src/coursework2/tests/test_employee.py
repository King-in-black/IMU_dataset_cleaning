# This file is deliberately badly written and does not conform to good practice for test case naming standards
import pytest

from src.coursework2.employee import Employee


def testemp1():
    e = Employee(name="A N Other", title="Manager", employee_id="12345", salary=45000)
    assert e.department == "HR"


def testemp2():
    e = Employee(name="A N Other", title="Manager", employee_id="12345", salary=45000)
    result = e.calculate_monthly_salary(hours_worked=155)
    assert result == 3875


def testemp3():
    e = Employee(name="A N Other", title="Manager", employee_id="12345", salary=45000)
    e.salary = 9000
    assert e.salary == 9000


def testemp4():
    e = Employee(name="A N Other", title="Manager", employee_id="12345", salary=45000)
    e.salary = 19575.67
    assert e.salary == 19575.67

