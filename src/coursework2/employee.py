# Sample code for testing. Docstrings ommitted.

class Employee:

    def __init__(self, name, employee_id, title, salary):
        self.name = name
        self.employee_id = employee_id
        self.title = title
        self.department = ""
        self._salary = salary

    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, value):
        if value < 15000:
            raise ValueError('Salary must be at least 15000')
        else:
            self._salary = value

    def calculate_monthly_salary(self, hours_worked):
        """ Overtime is paid at a rate of double the salary"""
        overtime = 0
        if hours_worked > 150:
            overtime = hours_worked - 150
        monthly_salary = self.salary/12 + (overtime * ((self.salary/12) / 150))
        return monthly_salary

    def assign_department(self, emp_department):
        self.department = emp_department

    def __str__(self):
        return f'{self.name} , id={self.employee_id}, in {self.department} department, and is a {self.title}, ' \
               f'with a salary of {self.salary}.'
