class Person: 
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def get_info(self):
        return f"{self.name} ({self.age} years old)"
    
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    def get_info(self):
        return f"{super().get_info()}, Student ID: {self.student_id}"
    

    