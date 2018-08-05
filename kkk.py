#A class object is first created with a name
class Employee:
    pass

emp_1 = Employee()
emp_2 = Employee()

emp_1.first = 'Albert'
emp_1.last = 'Gonsalez'


#Instance based classes
class addition:
    x = 5
    z = 0
    #Define a self initializing instance for the class
    def __init__(self, int1, int2, int3):
        self.int1 = int1 #Points to the varible
        self.int2 = int2
        self.int3 = int3

    addition.z += 1
    #Function which belongs to the class
    def total(self):
        self.total = self.int1 * self.int2 * self.int3
        print(self.total)


#Call the class with instances in it
number1 = addition(5, 3, 6)
number2 = addition(6,  5 ,4)

#Calling the function on the instance of the class
number1.total()

#Gives info on the class instance
print(number1.__dict__)

number1.x = 8

addition.x
number1.x

addition.z

#Class inheritance (Inherits attributes from the class
#which was referred to)
class Developer(addition):
    add_value = 4

    #Define a new instance, similar to "addition" class instance
    def __init__(self, int1, int2, int3, prog_lang):
        #Call in parent class to attach to subclass
        addition.__init__(self, int1, int2, int3)
        self.prog_lang = prog_lang

class Manager(addition):

    def __init__(self, int1, int2, int3, employees=None):
        if employees is None:
            self.employees = []
        else:
            self.employees = employees


dev1 = Developer(5, 6, 7, 'Java')
print(dev1.int3)
print(dev1.prog_lang)
dev1.total()
